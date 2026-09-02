import { beforeEach, describe, expect, it, vi } from "vitest";
import packageJson from "../package.json";

const pipelineMock = vi.hoisted(() => vi.fn());
vi.mock("@huggingface/transformers", () => ({ pipeline: pipelineMock }));

import { createEmbeddingRuntime } from "./index.js";

const tensor = (
	rows: number[][],
	dims: number[] | undefined = [rows.length, rows[0]?.length ?? 0],
) => ({ data: new Float64Array(rows.flat()), dims });

describe("createEmbeddingRuntime", () => {
	beforeEach(() => pipelineMock.mockReset());

	it("pins the default model and exposes its runtime identity", async () => {
		const extractor = vi.fn().mockResolvedValue(tensor([[1, 2, 3]]));
		pipelineMock.mockResolvedValue(extractor);
		const model = "Xenova/bge-small-en-v1.5";
		const revision = "ea104dacec62c0de699686887e3f920caeb4f3e3";
		const runtime = await createEmbeddingRuntime({ model });

		expect(pipelineMock).toHaveBeenCalledWith("feature-extraction", model, {
			revision,
			device: "cpu",
			dtype: "fp32",
		});
		expect(runtime.identity).toEqual({
			package: "@huggingface/transformers",
			version: packageJson.dependencies["@huggingface/transformers"],
			model,
			revision,
			dtype: "fp32",
			device: "cpu",
			dimensions: 3,
		});
		expect(extractor).toHaveBeenCalledWith(["probe"], {
			pooling: "mean",
			normalize: true,
		});
	});

	it("requires a revision for a custom model", async () => {
		await expect(createEmbeddingRuntime({ model: "test/model" })).rejects.toThrow(
			"A revision is required when creating a runtime for a custom model",
		);
		expect(pipelineMock).not.toHaveBeenCalled();
	});

	it("uses the requested custom model revision", async () => {
		pipelineMock.mockResolvedValue(vi.fn().mockResolvedValue(tensor([[1, 2]])));
		const runtime = await createEmbeddingRuntime({ model: "test/model", revision: "release" });

		expect(pipelineMock).toHaveBeenCalledWith("feature-extraction", "test/model", {
			revision: "release",
			device: "cpu",
			dtype: "fp32",
		});
		expect(runtime.identity.revision).toBe("release");
	});

	it("honors an explicit revision for the default model", async () => {
		pipelineMock.mockResolvedValue(vi.fn().mockResolvedValue(tensor([[1, 2]])));
		const model = "Xenova/bge-small-en-v1.5";
		const revision = "release-override";

		const runtime = await createEmbeddingRuntime({ model, revision });

		expect(pipelineMock).toHaveBeenCalledWith("feature-extraction", model, {
			revision,
			device: "cpu",
			dtype: "fp32",
		});
		expect(runtime.identity.revision).toBe(revision);
	});

	it("batches at 32 while preserving order and returning owned fp32 rows", async () => {
		const calls: string[][] = [];
		const outputs: Float64Array[] = [];
		const extractor = vi.fn(async (texts: string[]) => {
			calls.push(texts);
			const output =
				texts[0] === "probe"
					? tensor([[0, 0]])
					: tensor(texts.map((text) => [Number(text), Number(text) + 0.5]));
			outputs.push(output.data);
			return output;
		});
		pipelineMock.mockResolvedValue(extractor);
		const runtime = await createEmbeddingRuntime({
			model: "test/model",
			revision: "test-revision",
		});
		const vectors = await runtime.embed(Array.from({ length: 65 }, (_, index) => String(index)));

		expect(calls.slice(1).map((call) => call.length)).toEqual([32, 32, 1]);
		expect(vectors.map((row) => [...row])).toEqual(
			Array.from({ length: 65 }, (_, index) => [index, index + 0.5]),
		);
		const firstBuffer = vectors[0]?.buffer;
		expect(vectors.every((row, index) => index === 0 || row.buffer !== firstBuffer)).toBe(true);
		expect(vectors.every((row) => outputs.every((output) => row.buffer !== output.buffer))).toBe(
			true,
		);
	});

	it("does not invoke inference for empty input", async () => {
		const extractor = vi.fn().mockResolvedValue(tensor([[0, 0]]));
		pipelineMock.mockResolvedValue(extractor);
		const runtime = await createEmbeddingRuntime({
			model: "test/model",
			revision: "test-revision",
		});
		extractor.mockClear();

		await expect(runtime.embed([])).resolves.toEqual([]);
		expect(extractor).not.toHaveBeenCalled();
	});

	it.each([
		{ data: new Float32Array([1, 2]), dims: [2, 1] },
		{ data: new Float32Array([1]), dims: [1, 2] },
		{ data: new Float32Array(384), dims: undefined },
		{ data: new Float32Array([1, Number.NaN]), dims: [1, 2] },
	])("rejects malformed tensors", async (probe) => {
		pipelineMock.mockResolvedValue(vi.fn().mockResolvedValue(probe));
		await expect(
			createEmbeddingRuntime({ model: "bad/model", revision: "test-revision" }),
		).rejects.toThrow(TypeError);
	});

	it.each([
		{ data: new Float32Array([1]), dims: [1, 1] },
		{ data: new Float32Array([1, 2]), dims: undefined },
		{ data: new Float32Array([1, Number.NaN]), dims: [1, 2] },
	])("rejects malformed inference tensors", async (output) => {
		const extractor = vi
			.fn()
			.mockResolvedValueOnce(tensor([[0, 0]]))
			.mockResolvedValueOnce(output);
		pipelineMock.mockResolvedValue(extractor);
		const runtime = await createEmbeddingRuntime({
			model: "bad/model",
			revision: "test-revision",
		});

		await expect(runtime.embed(["text"])).rejects.toThrow(TypeError);
	});
});
