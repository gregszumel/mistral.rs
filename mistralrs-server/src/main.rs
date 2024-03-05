use std::{
    collections::HashMap,
    fs::File,
    sync::{mpsc::channel, Arc},
};

use anyhow::Result;
use axum::{
    extract::{Json, State},
    routing::post,
    Router,
};
use candle_core::Device;
use clap::{Parser, Subcommand};
use mistralrs_core::{
    Conversation, GemmaLoader, GemmaSpecificConfig, LlamaLoader, LlamaSpecificConfig, Loader,
    MistralLoader, MistralRs, MistralSpecificConfig, ModelKind, Request, Response, SamplingParams,
    SchedulerMethod, StopTokens as InternalStopTokens, TokenSource,
};
use openai::{ChatCompletionRequest, StopTokens};
mod openai;

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the mistral model.
    Mistral {
        /// Model ID to load from
        #[arg(short, long, default_value = "mistralai/Mistral-7B-Instruct-v0.1")]
        model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized mistral model with gguf.
    MistralGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "mistralai/Mistral-7B-Instruct-v0.1")]
        tok_model_id: String,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        #[arg(
            short = 'm',
            long,
            default_value = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        )]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the mistral model, with X-LoRA.
    XLoraMistral {
        /// Model ID to load from
        #[arg(short, long, default_value = "HuggingFaceH4/zephyr-7b-beta")]
        model_id: String,

        /// Model ID to load Xlora from
        #[arg(short, long, default_value = "lamm-mit/x-lora")]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the gemma model.
    Gemma {
        /// Model ID to load from
        #[arg(short, long, default_value = "google/gemma-7b-it")]
        model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the gemma model, with X-LoRA.
    XLoraGemma {
        /// Model ID to load from
        #[arg(short, long)]
        model_id: String,

        /// Model ID to load Xlora from
        #[arg(short, long)]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the llama model.
    Llama {
        /// Model ID to load from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized llama model with gguf.
    LlamaGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        tok_model_id: String,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "llama-2-13b-chat.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized llama model with gguf.
    LlamaGGML {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        tok_model_id: String,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGML")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "llama-2-13b-chat.ggmlv3.q4_K_M.bin"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// GQA
        #[arg(long, default_value_t = 1)]
        gqa: usize,
    },
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Port to serve on.
    #[arg(short, long)]
    port: String,

    /// Log all responses and requests to this file
    #[clap(long, short)]
    log: Option<String>,

    /// If a sequence is larger than the maximum model length, truncate the number
    /// of tokens such that the sequence will fit at most the maximum length.
    /// If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.
    #[clap(long, short, action)]
    truncate_sequence: bool,

    /// Model
    #[clap(subcommand)]
    model: ModelSelected,

    /// Maximum running sequences at any time
    #[arg(long, default_value_t = 2)]
    max_seqs: usize,

    /// Use no KV cache for X-LoRA, only applicable for X-LoRA models and the Llama Normal model.
    #[arg(long, default_value_t = false)]
    no_xlora_kv_cache: bool,
}

async fn chatcompletions(
    State((state, conv)): State<(Arc<MistralRs>, Arc<dyn Conversation + Send + Sync>)>,
    Json(oairequest): Json<ChatCompletionRequest>,
) -> String {
    let (tx, rx) = channel();
    let repr = serde_json::to_string(&oairequest).unwrap();
    let stop_toks = match oairequest.stop_seqs {
        Some(StopTokens::Multi(m)) => Some(InternalStopTokens::Seqs(m)),
        Some(StopTokens::Single(s)) => Some(InternalStopTokens::Seqs(vec![s])),
        Some(StopTokens::MultiId(m)) => Some(InternalStopTokens::Ids(m)),
        Some(StopTokens::SingleId(s)) => Some(InternalStopTokens::Ids(vec![s])),
        None => None,
    };
    let mut messages = Vec::new();
    for message in oairequest.messages {
        let mut message_map = HashMap::new();
        message_map.insert("role".to_string(), message.role);
        message_map.insert("content".to_string(), message.content);
        messages.push(message_map);
    }
    let prompt = match conv.get_prompt(messages, true) {
        Err(e) => return e,
        Ok(p) => p,
    };
    let request = Request {
        prompt,
        sampling_params: SamplingParams {
            temperature: oairequest.temperature,
            top_k: oairequest.top_k,
            top_p: oairequest.top_p,
            top_n_logprobs: oairequest.top_logprobs.unwrap_or(1),
            repeat_penalty: oairequest.repetition_penalty,
            presence_penalty: oairequest.presence_penalty,
            max_len: oairequest.max_tokens,
            stop_toks,
        },
        response: tx,
        return_logprobs: oairequest.logprobs,
    };

    MistralRs::maybe_log_request(state.clone(), repr);
    let sender = state.get_sender();
    sender.send(request).unwrap();
    let response = rx.recv().unwrap();

    match response {
        Response::Error(e) => {
            dbg!(&e);
            e.to_string()
        }
        Response::Done(response) => {
            MistralRs::maybe_log_response(state, &response);
            serde_json::to_string(&response).unwrap()
        }
    }
}

fn get_router(state: (Arc<MistralRs>, Arc<dyn Conversation + Send + Sync>)) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chatcompletions))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let loader: Box<dyn Loader> = match args.model {
        ModelSelected::Mistral {
            model_id,
            repeat_last_n,
        } => Box::new(MistralLoader::new(
            model_id,
            MistralSpecificConfig {
                use_flash_attn: false,
                repeat_last_n,
            },
            None,
            None,
            None,
            ModelKind::Normal,
            None,
            args.no_xlora_kv_cache,
        )),
        ModelSelected::MistralGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
        } => Box::new(MistralLoader::new(
            tok_model_id,
            MistralSpecificConfig {
                use_flash_attn: false,
                repeat_last_n,
            },
            quantized_model_id,
            quantized_filename,
            None,
            ModelKind::QuantizedGGUF,
            None,
            args.no_xlora_kv_cache,
        )),
        ModelSelected::XLoraMistral {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
        } => Box::new(MistralLoader::new(
            model_id,
            MistralSpecificConfig {
                use_flash_attn: false,
                repeat_last_n,
            },
            None,
            None,
            Some(xlora_model_id),
            ModelKind::XLoraNormal,
            Some(serde_json::from_reader(File::open(order)?)?),
            args.no_xlora_kv_cache,
        )),
        ModelSelected::Gemma {
            model_id,
            repeat_last_n,
        } => Box::new(GemmaLoader::new(
            model_id,
            GemmaSpecificConfig { repeat_last_n },
            None,
            None,
            None,
            ModelKind::Normal,
            None,
            args.no_xlora_kv_cache,
        )),
        ModelSelected::XLoraGemma {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
        } => Box::new(GemmaLoader::new(
            model_id,
            GemmaSpecificConfig { repeat_last_n },
            None,
            None,
            Some(xlora_model_id),
            ModelKind::Normal,
            Some(serde_json::from_reader(File::open(order)?)?),
            args.no_xlora_kv_cache,
        )),
        ModelSelected::Llama {
            model_id,
            repeat_last_n,
        } => Box::new(LlamaLoader::new(
            model_id,
            LlamaSpecificConfig {
                repeat_last_n,
                use_flash_attn: false,
                gqa: 0,
            },
            None,
            None,
            None,
            ModelKind::Normal,
            None,
            args.no_xlora_kv_cache,
        )),
        ModelSelected::LlamaGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
        } => Box::new(LlamaLoader::new(
            tok_model_id,
            LlamaSpecificConfig {
                repeat_last_n,
                use_flash_attn: false,
                gqa: 0,
            },
            quantized_model_id,
            quantized_filename,
            None,
            ModelKind::QuantizedGGUF,
            None,
            args.no_xlora_kv_cache,
        )),
        ModelSelected::LlamaGGML {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            gqa,
        } => Box::new(LlamaLoader::new(
            tok_model_id,
            LlamaSpecificConfig {
                repeat_last_n,
                use_flash_attn: false,
                gqa,
            },
            quantized_model_id,
            quantized_filename,
            None,
            ModelKind::QuantizedGGML,
            None,
            args.no_xlora_kv_cache,
        )),
    };

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::cuda_if_available(0)?;

    let (pipeline, conv) = loader.load_model(None, TokenSource::CacheToken, None, &device)?;
    let mistralrs = MistralRs::new(
        pipeline,
        SchedulerMethod::Fixed(args.max_seqs.try_into().unwrap()),
        args.log,
        args.truncate_sequence,
    );

    let app = get_router((mistralrs, conv));

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.port)).await?;
    axum::serve(listener, app).await?;

    Ok(())
}