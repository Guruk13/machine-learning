/* tslint:disable */
/* eslint-disable */

export enum Action {
    DoNothing = 0,
    Jump = 1,
}

/**
 * In ML, gamma (γ) is the discount factor used in reinforcement learning (RL).
 * * It's a value between 0 and 1 that determines how much an agent values future rewards relative to immediate rewards.
 * The Bellman Equation Context
 * Gamma appears in the return (cumulative reward) calculation:
 * G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
 * What the Value Means
 * GammaBehaviorγ = 0 Fully myopic — only cares about the immediate next rewardγ →
 * 1Fully far-sighted — values future rewards almost as much as current onesγ = 0.99 Common default — slight preference for sooner rewards
 */
export class AgentDefault {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    gamma: number;
    learning_rate: number;
}

export class FlappyGradientAgent {
    free(): void;
    [Symbol.dispose](): void;
    agent_clear_episode(): void;
    agent_finish_episode(): void;
    constructor(settings: AgentDefault);
    new_model_reset_stats(): void;
    record_step(action: Action, reward: number): void;
    /**
     * Sync, non-blocking. Call this once per frame from a Bevy system.
     *
     * Returns the most recently completed inference result (may lag a
     * frame or two behind `state`), and opportunistically kicks off a new
     * GPU inference in the background if one isn't already running.
     */
    select_action(): Action;
    set_model(net: FlappyNet): void;
    /**
     * Call when the bird dies.
     *
     * Computes discounted returns, normalises them, runs the forward pass,
     * computes the REINFORCE loss, backprops, and steps the optimizer — all
     * in one function. Uses burn's async tensor API (`into_scalar_async`)
     * so it's WASM-safe; the caller is responsible for awaiting it (e.g.
     * from an async-exported wasm_bindgen method, or from within a task if
     * you add one later).
     *
     * Returns the scalar loss for this episode.
     */
    update_policy(): Promise<number>;
}

export class FlappyNet {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
}

/**
 * Format that each sample has. Usually, this corresponds to the sampling
 * depth of the audio source. For example, 16 bit quantized samples can be
 * encoded in `i16` or `u16`. Note that the quantized sampling depth is not
 * directly visible for formats where [`is_float`] is true.
 *
 * Also note that the backend must support the encoding of the quantized
 * samples in the given format, as there is no generic transformation from one
 * format into the other done inside the frontend-library code. You can query
 * the supported formats by using [`supported_input_configs`].
 *
 * A good rule of thumb is to use [`SampleFormat::I16`] as this covers typical
 * music (WAV, MP3) as well as typical audio input devices on most platforms,
 *
 * [`is_float`]: SampleFormat::is_float
 * [`supported_input_configs`]: crate::traits::DeviceTrait::supported_input_configs
 */
export enum SampleFormat {
    /**
     * `i8` with a valid range of `i8::MIN..=i8::MAX` with `0` being the origin.
     */
    I8 = 0,
    /**
     * `i16` with a valid range of `i16::MIN..=i16::MAX` with `0` being the origin.
     */
    I16 = 1,
    /**
     * `I24` with a valid range of `-(1 << 23)..=((1 << 23) - 1)` with `0` being the origin.
     *
     * This format uses 4 bytes of storage but only 24 bits are significant.
     */
    I24 = 2,
    /**
     * `i32` with a valid range of `i32::MIN..=i32::MAX` with `0` being the origin.
     */
    I32 = 3,
    /**
     * `i64` with a valid range of `i64::MIN..=i64::MAX` with `0` being the origin.
     */
    I64 = 4,
    /**
     * `u8` with a valid range of `u8::MIN..=u8::MAX` with `1 << 7 == 128` being the origin.
     */
    U8 = 5,
    /**
     * `u16` with a valid range of `u16::MIN..=u16::MAX` with `1 << 15 == 32768` being the origin.
     */
    U16 = 6,
    /**
     * `U24` with a valid range of `0..=((1 << 24) - 1)` with `1 << 23 == 8388608` being the origin.
     *
     * This format uses 4 bytes of storage but only 24 bits are significant.
     */
    U24 = 7,
    /**
     * `u32` with a valid range of `u32::MIN..=u32::MAX` with `1 << 31` being the origin.
     */
    U32 = 8,
    /**
     * `U48` with a valid range of '0..(1 << 48)' with `1 << 47` being the origin
     * `u64` with a valid range of `u64::MIN..=u64::MAX` with `1 << 63` being the origin.
     */
    U64 = 9,
    /**
     * `f32` with a valid range of `-1.0..=1.0` with `0.0` being the origin.
     */
    F32 = 10,
    /**
     * `f64` with a valid range of `-1.0..=1.0` with `0.0` being the origin.
     */
    F64 = 11,
    /**
     * DSD 1-bit stream in u8 container (8 bits = 8 DSD samples) with 0x69 being the silence byte pattern.
     */
    DsdU8 = 12,
    /**
     * DSD 1-bit stream in u16 container (16 bits = 16 DSD samples) with 0x69 being the silence byte pattern.
     */
    DsdU16 = 13,
    /**
     * DSD 1-bit stream in u32 container (32 bits = 32 DSD samples) with 0x69 being the silence byte pattern.
     */
    DsdU32 = 14,
}

/**
 * The set of parameters used to describe how to open a stream.
 *
 * The sample format is omitted in favour of using a sample type.
 *
 * See also [`BufferSize`] for details on buffer size behavior and latency considerations.
 */
export class StreamConfig {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get buffer_size(): number | undefined;
    set buffer_size(value: number | null | undefined);
    channels: number;
    sample_rate: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly main: (a: number, b: number) => number;
    readonly __wbg_agentdefault_free: (a: number, b: number) => void;
    readonly __wbg_flappygradientagent_free: (a: number, b: number) => void;
    readonly __wbg_flappynet_free: (a: number, b: number) => void;
    readonly __wbg_get_agentdefault_gamma: (a: number) => number;
    readonly __wbg_get_agentdefault_learning_rate: (a: number) => number;
    readonly __wbg_get_streamconfig_buffer_size: (a: number) => number;
    readonly __wbg_get_streamconfig_channels: (a: number) => number;
    readonly __wbg_get_streamconfig_sample_rate: (a: number) => number;
    readonly __wbg_set_agentdefault_gamma: (a: number, b: number) => void;
    readonly __wbg_set_agentdefault_learning_rate: (a: number, b: number) => void;
    readonly __wbg_set_streamconfig_buffer_size: (a: number, b: number) => void;
    readonly __wbg_set_streamconfig_channels: (a: number, b: number) => void;
    readonly __wbg_set_streamconfig_sample_rate: (a: number, b: number) => void;
    readonly __wbg_streamconfig_free: (a: number, b: number) => void;
    readonly flappygradientagent_agent_clear_episode: (a: number) => void;
    readonly flappygradientagent_agent_finish_episode: (a: number) => void;
    readonly flappygradientagent_new: (a: number) => number;
    readonly flappygradientagent_new_model_reset_stats: (a: number) => void;
    readonly flappygradientagent_record_step: (a: number, b: number, c: number) => void;
    readonly flappygradientagent_select_action: (a: number) => number;
    readonly flappygradientagent_set_model: (a: number, b: number) => void;
    readonly flappygradientagent_update_policy: (a: number) => any;
    readonly wasm_bindgen__convert__closures_____invoke__h3e0812ddd4a7f2b6: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h62fa7dbba94fe9f7: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h723db84ca0cdcf70: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1f0eae7f2f82e99d: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1f0eae7f2f82e99d_24: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1f0eae7f2f82e99d_25: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1f0eae7f2f82e99d_26: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1f0eae7f2f82e99d_28: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1f0eae7f2f82e99d_29: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1f0eae7f2f82e99d_30: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h1f0eae7f2f82e99d_31: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h612fe9021a1297eb: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h612fe9021a1297eb_27: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h697729f5d7143ab4: (a: number, b: number, c: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h87c1eea3bd138b93: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h7d19657fba27c1dd: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hb9ad98521667a024: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
