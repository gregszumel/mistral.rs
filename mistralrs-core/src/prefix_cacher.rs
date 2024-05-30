use std::{
    ops::Deref,
    sync::{Arc, Mutex},
};

use candle_core::{Device, Result, Tensor};
use radix_trie::{Trie, TrieCommon, TrieKey};

use crate::{get_mut_arcmutex, pipeline::LayerCaches, sequence::Sequence};

#[derive(PartialEq, Eq)]
struct Tokens(Vec<u32>);

impl TrieKey for Tokens {
    fn encode_bytes(&self) -> Vec<u8> {
        self.0
            .iter()
            .flat_map(|x| bytemuck::bytes_of(x).to_vec())
            .collect::<Vec<u8>>()
    }
}

impl From<Vec<u32>> for Tokens {
    fn from(value: Vec<u32>) -> Self {
        Self(value)
    }
}

type EvictionCacheGroup = (Arc<Mutex<LayerCaches>>, Option<Arc<Mutex<LayerCaches>>>);

pub struct PrefixCacheManager {
    caches: Trie<Tokens, Arc<Mutex<LayerCaches>>>,
    xlora_caches: Option<Trie<Tokens, Arc<Mutex<LayerCaches>>>>,
    device: Device,
    pub total_bytes_limit: usize,
    pub device_bytes_limit: usize,
    pub total_bytes: usize,
    pub device_bytes: usize,
    no_prefix_cache: bool,
    eviction_cache_ptrs: Vec<EvictionCacheGroup>,
}

#[derive(Clone)]
pub struct MatchingCache {
    pub normal: LayerCaches,
    pub xlora: Option<LayerCaches>,
    pub toks: Vec<u32>,
}

impl PrefixCacheManager {
    pub fn new(device: Device, n_on_device: usize, is_xlora: bool, no_prefix_cache: bool) -> Self {
        PrefixCacheManager {
            caches: Trie::new(),
            xlora_caches: if is_xlora { Some(Trie::new()) } else { None },
            device,
            device_bytes_limit: 2_000_000,
            total_bytes_limit: 1_000_000,
            device_bytes: 0,
            total_bytes: 0,
            no_prefix_cache,
            eviction_cache_ptrs: Vec::new(),
        }
    }

    /// Get's the number of bytes in a KV cache.
    fn get_kv_cache_byte_size(cache: &Vec<Option<(Tensor, Tensor)>>) -> usize {
        cache
            .iter()
            .map(|pair| match pair {
                Some((t1, t2)) => {
                    let t1_bytes = t1.elem_count() * t1.dtype().size_in_bytes();
                    let t2_bytes = t2.elem_count() * t2.dtype().size_in_bytes();
                    t1_bytes + t2_bytes
                }
                None => 0,
            })
            .sum()
    }

    /// This always keeps the cache on the device. If later on, a new seq cannot be allocated due to memory shortage,
    /// some caches will be evicted.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        if self.no_prefix_cache {
            return;
        }
        let cache = Arc::new(Mutex::new(seq.cache().clone()));

        // count bytes in kv cache. TODO: Can we cache this per model?
        let cache_bytes = PrefixCacheManager::get_kv_cache_byte_size(seq.cache());

        self.total_bytes += cache_bytes;
        let device = seq.cache().deref()[0].as_ref().unwrap().0.device().clone();
        let bytes_on_device_increment = match &device {
            Device::Cpu => 0,
            _non_cpu => cache_bytes,
        };
        self.device_bytes += bytes_on_device_increment;

        self.caches
            .insert(seq.get_toks().to_vec().into(), cache.clone());
        if seq.is_xlora() {
            let xlora_cache = Arc::new(Mutex::new(seq.xlora_cache().clone()));
            let xlora_cache_bytes = PrefixCacheManager::get_kv_cache_byte_size(seq.xlora_cache());
            self.total_bytes += xlora_cache_bytes;
            self.device_bytes += match &device {
                Device::Cpu => 0,
                _else => xlora_cache_bytes,
            };

            self.xlora_caches
                .as_mut()
                .unwrap()
                .insert(seq.get_toks().to_vec().into(), xlora_cache.clone());
            self.eviction_cache_ptrs.push((cache, Some(xlora_cache)));
        } else {
            self.eviction_cache_ptrs.push((cache, None));
        }
    }

    fn cache_to<'a>(
        cache: impl Iterator<Item = &'a mut Option<(Tensor, Tensor)>>,
        device: &Device,
    ) -> Result<()> {
        for layer in cache {
            if let Some((ref q, ref k)) = layer {
                *layer = Some((q.to_device(device)?, k.to_device(device)?));
            }
        }
        Ok(())
    }

    /// Evict the caches to CPU. This will evict the first k seqs such that the number of sequences on device after the copy is
    /// the maximum allowed. Returns the number of evicted sequences.
    pub fn evict_to_cpu(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        // Intentionally evict the first ones first, as they are the oldest
        for (cache, xlora_cache) in &self.eviction_cache_ptrs {
            if self.device_bytes <= self.device_bytes_limit {
                break;
            }
            if !matches!(
                get_mut_arcmutex!(cache.as_ref())[0]
                    .as_ref()
                    .unwrap()
                    .0
                    .device(),
                Device::Cpu
            ) {
                let mut cache = get_mut_arcmutex!(cache);
                let mut xlora_cache = xlora_cache.as_ref().map(|c| get_mut_arcmutex!(c));
                let cache_size = PrefixCacheManager::get_kv_cache_byte_size(&mut cache);
                let xlora_cache_size = if let Some(ref valid_cache) = xlora_cache {
                    PrefixCacheManager::get_kv_cache_byte_size(&valid_cache)
                } else {
                    0
                };

                Self::cache_to(cache.iter_mut(), &Device::Cpu)?;
                if let Some(ref mut xlora_cache) = xlora_cache {
                    Self::cache_to(xlora_cache.iter_mut(), &Device::Cpu)?;
                }
                self.device_bytes -= cache_size + xlora_cache_size
            }
        }
        Ok(self.total_bytes - self.device_bytes)
    }

    /// Evict all the caches to CPU.
    pub fn evict_all_to_cpu(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        // Intentionally evict the first ones first, as they are the oldest
        for (cache, xlora_cache) in &self.eviction_cache_ptrs {
            if !matches!(
                get_mut_arcmutex!(cache.as_ref())[0]
                    .as_ref()
                    .unwrap()
                    .0
                    .device(),
                Device::Cpu
            ) {
                let mut cache = get_mut_arcmutex!(cache);
                let mut xlora_cache = xlora_cache.as_ref().map(|c| get_mut_arcmutex!(c));

                Self::cache_to(cache.iter_mut(), &Device::Cpu)?;
                if let Some(ref mut xlora_cache) = xlora_cache {
                    Self::cache_to(xlora_cache.iter_mut(), &Device::Cpu)?;
                }
            }
        }
        self.device_bytes = 0;
        Ok(self.caches.len())
    }

    /// Search for a matching cache given some toks
    pub fn search_for_matching_cache(&mut self, toks: &[u32]) -> Result<Option<MatchingCache>> {
        if self.no_prefix_cache {
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());
        if let Some(cache) = self.caches.get(&toks) {
            Self::cache_to(get_mut_arcmutex!(cache.as_ref()).iter_mut(), &self.device)?;
            let cache = get_mut_arcmutex!(cache.as_ref()).clone();
            let xlora_cache = if let Some(ref xlora_caches) = self.xlora_caches {
                let mut xlora_cache = get_mut_arcmutex!(xlora_caches.get(&toks).unwrap().as_ref());
                Self::cache_to(xlora_cache.iter_mut(), &self.device)?;
                Some(xlora_cache.clone())
            } else {
                None
            };
            let ancestor = &self
                .caches
                .get_ancestor(&toks)
                .expect("No ancestor.")
                .key()
                .expect("Cannot get the key.")
                .0;
            // Know ancestor.len() < toks.len(), and toks[0..ancestor.len()] == toks
            Ok(Some(MatchingCache {
                normal: cache,
                xlora: xlora_cache,
                toks: toks.0[ancestor.len()..].to_vec(),
            }))
        } else {
            Ok(None)
        }
    }

    // get rid of all the cache
    pub fn purge_cache() {}

    // trim the cache size back down to below threshold
    pub fn trim_cache() {}
}
