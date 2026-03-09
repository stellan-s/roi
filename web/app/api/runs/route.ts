import { NextRequest, NextResponse } from "next/server";
import { RunMode, RunStatus } from "@prisma/client";

import { prisma } from "@/lib/prisma";
import { runQuant, type QuantMode } from "@/lib/quant";

function parseMode(value: unknown): QuantMode {
  if (value === "adaptive") return "adaptive";
  return "static";
}

export async function GET() {
  const runs = await prisma.run.findMany({
    orderBy: { createdAt: "desc" },
    take: 25,
    include: {
      _count: { select: { recommendations: true } },
    },
  });
  return NextResponse.json({ runs });
}

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}));
  const mode = parseMode(body?.mode);

  const created = await prisma.run.create({
    data: {
      mode: mode === "adaptive" ? RunMode.ADAPTIVE : RunMode.STATIC,
      status: RunStatus.RUNNING,
      startedAt: new Date(),
    },
  });

  try {
    const result = await runQuant(mode);

    await prisma.$transaction(async (tx) => {
      await tx.recommendation.createMany({
        data: result.recommendations.map((rec) => ({
          runId: created.id,
          date: rec.date ? new Date(rec.date) : null,
          ticker: rec.ticker,
          decision: rec.decision,
          close: rec.close,
          expectedReturn: rec.expected_return,
          probPositive: rec.prob_positive,
          decisionConfidence: rec.decision_confidence,
          uncertainty: rec.uncertainty,
          portfolioWeight: rec.portfolio_weight,
          auditJson: rec.audit ? JSON.stringify(rec.audit) : null,
        })),
      });

      await tx.run.update({
        where: { id: created.id },
        data: {
          status: RunStatus.COMPLETED,
          endedAt: new Date(),
          summaryJson: JSON.stringify(result.summary),
        },
      });
    });

    const hydrated = await prisma.run.findUnique({
      where: { id: created.id },
      include: {
        _count: { select: { recommendations: true } },
      },
    });

    return NextResponse.json({ run: hydrated });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    await prisma.run.update({
      where: { id: created.id },
      data: {
        status: RunStatus.FAILED,
        endedAt: new Date(),
        error: message,
      },
    });

    return NextResponse.json(
      { error: "Run failed", details: message, runId: created.id },
      { status: 500 },
    );
  }
}
