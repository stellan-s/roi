import { spawn } from "node:child_process";
import path from "node:path";

export type QuantMode = "static" | "adaptive";

export type QuantBridgeResult = {
  mode: string;
  started_at_utc: string;
  ended_at_utc: string;
  summary: Record<string, unknown>;
  recommendations: Array<{
    date: string | null;
    ticker: string;
    decision: string;
    close: number | null;
    expected_return: number | null;
    prob_positive: number | null;
    decision_confidence: number | null;
    uncertainty: number | null;
    portfolio_weight: number | null;
    rationale?: string | null;
    audit?: Record<string, unknown> | null;
  }>;
};

function resolvePythonBin(): string {
  if (process.env.ROI_PYTHON_BIN) {
    const configured = process.env.ROI_PYTHON_BIN;
    return path.isAbsolute(configured)
      ? configured
      : path.resolve(process.cwd(), configured);
  }
  return path.resolve(process.cwd(), "../.venv/bin/python");
}

function parseJsonFromStdout(stdout: string): QuantBridgeResult {
  const lines = stdout
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    throw new Error("Quant bridge returned no stdout payload");
  }

  // Bridge prints JSON on the final line.
  const payload = lines[lines.length - 1];
  return JSON.parse(payload) as QuantBridgeResult;
}

export async function runQuant(mode: QuantMode): Promise<QuantBridgeResult> {
  const pythonBin = resolvePythonBin();
  const args = ["-m", "quant.bridge.run_once", "--mode", mode];
  if (process.env.ROI_QUANT_DRY_RUN === "1") {
    args.push("--dry-run");
  }

  return await new Promise<QuantBridgeResult>((resolve, reject) => {
    const child = spawn(pythonBin, args, {
      cwd: path.resolve(process.cwd(), ".."),
      env: process.env,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (err) => {
      reject(err);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Quant process failed with code ${code}: ${stderr}`));
        return;
      }
      try {
        const parsed = parseJsonFromStdout(stdout);
        resolve(parsed);
      } catch (err) {
        reject(
          new Error(
            `Failed to parse quant bridge output: ${(err as Error).message}. stderr=${stderr}`,
          ),
        );
      }
    });
  });
}
