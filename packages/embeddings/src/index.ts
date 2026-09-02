export interface EmbeddingRuntimeRequest {
	model: string;
	revision?: string;
}

export interface EmbeddingRuntimeIdentity {
	readonly package: "@huggingface/transformers";
	readonly version: string;
	readonly model: string;
	readonly revision: string;
	readonly dtype: "fp32";
	readonly device: "cpu";
	readonly dimensions: number;
}

export interface EmbeddingClient {
	readonly model: string;
	readonly dimensions: number;
	readonly identity: EmbeddingRuntimeIdentity;
	embed(texts: string[]): Promise<Float32Array[]>;
}

export type EmbeddingRuntimeFactory = (
	request: EmbeddingRuntimeRequest,
) => Promise<EmbeddingClient | null>;

interface Tensor {
	data: ArrayLike<number | bigint>;
	dims?: number[];
}

type Extractor = (
	texts: string[],
	options: { pooling: "mean"; normalize: true },
) => Promise<Tensor>;

const DEFAULT_MODEL = "Xenova/bge-small-en-v1.5";
const DEFAULT_REVISION = "ea104dacec62c0de699686887e3f920caeb4f3e3";

function rows(output: Tensor, count: number, dimensions: number): Float32Array[] {
	if (
		output.dims?.length !== 2 ||
		(output.dims?.length === 2 && (output.dims[0] !== count || output.dims[1] !== dimensions)) ||
		output.data.length !== count * dimensions
	) {
		throw new TypeError("Embedding model returned an invalid tensor shape or value count");
	}
	const result = Array.from({ length: count }, () => new Float32Array(dimensions));
	for (let index = 0; index < output.data.length; index++) {
		const value = output.data[index];
		const number = typeof value === "number" ? Math.fround(value) : Number.NaN;
		if (!Number.isFinite(number)) {
			throw new TypeError(`Embedding model returned non-finite data at index ${index}`);
		}
		const row = result[Math.floor(index / dimensions)];
		if (!row) throw new TypeError("Embedding model returned an invalid tensor row");
		row[index % dimensions] = number;
	}
	return result;
}

export async function createEmbeddingRuntime({
	model,
	revision: requestedRevision,
}: EmbeddingRuntimeRequest): Promise<EmbeddingClient> {
	const configuredRevision = requestedRevision?.trim();
	if (model !== DEFAULT_MODEL && !configuredRevision) {
		throw new TypeError("A revision is required when creating a runtime for a custom model");
	}
	const revision = configuredRevision || DEFAULT_REVISION;
	const { pipeline } = await import("@huggingface/transformers");
	// Runtime validation below guards the structural boundary hidden by upstream's broad pipeline type.
	const extractor = (await pipeline("feature-extraction", model, {
		revision,
		device: "cpu",
		dtype: "fp32",
	})) as unknown as Extractor;
	const options = { pooling: "mean", normalize: true } as const;
	const probe = await extractor(["probe"], options);
	const dimensions = probe.dims?.at(-1);
	if (typeof dimensions !== "number" || !Number.isInteger(dimensions) || dimensions <= 0) {
		throw new TypeError("Embedding model returned invalid dimensions");
	}
	rows(probe, 1, dimensions);

	return {
		model,
		dimensions,
		identity: {
			package: "@huggingface/transformers",
			version: "4.2.0",
			model,
			revision,
			dtype: "fp32",
			device: "cpu",
			dimensions,
		},
		async embed(texts) {
			const result: Float32Array[] = [];
			for (let offset = 0; offset < texts.length; offset += 32) {
				const batch = texts.slice(offset, offset + 32);
				result.push(...rows(await extractor(batch, options), batch.length, dimensions));
			}
			return result;
		},
	};
}
