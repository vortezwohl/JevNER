/**
 * JevNER's underlying API, in plain TypeScript — zero dependencies, Node 18+.
 *
 * A toy NER loop over one sentence, showing the wire protocol the Python SDK
 * is built on:
 *   1. Noul question per word       -> "is this word part of an entity?" (probability)
 *   2. Choice question per hit      -> "which entity type is it?" (calibrated distribution)
 *
 * Run:  npx tsx ner.ts        (or compile with tsc and run with node)
 * Env:  TYPESAFE_API_KEY (required), TYPESAFE_BASE_URL (default https://api.easyjev.online)
 *
 * This is a learning example for the protocol. Real NER needs windowing,
 * byte-budget safety, caching, taxonomy and nested-tree assembly — that is
 * what the `jevner` Python package does.
 */

const BASE_URL = process.env.TYPESAFE_BASE_URL ?? "https://api.easyjev.online";
const API_KEY = process.env.TYPESAFE_API_KEY;
const MODEL = process.env.JEVNER_MODEL ?? "baseten/mammoth-1";

/** One typed question. `type` decides the answer shape. */
type Question =
  | { type: "noul"; instructions: unknown; criteria: { true: string; false: string } }
  | { type: "choice"; instructions: unknown; criteria: Record<string, string> };

interface NoulAnswer {
  type: "noul";
  /** Probability of "yes", calibrated by the decision model. */
  noul: number;
}

interface ChoiceAnswer {
  type: "choice";
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface SystemOneResponse {
  model: string;
  answers: Record<string, NoulAnswer | ChoiceAnswer>;
  usage: Record<string, number>;
}

/** POST one batch of typed questions against shared state. */
async function systemOne(state: unknown, questions: Record<string, Question>): Promise<SystemOneResponse> {
  const response = await fetch(`${BASE_URL}/v1/systemone`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      "Content-Type": "application/json",
      // The gateway rejects requests without a User-Agent.
      "User-Agent": "jevner-example/0.1 (+https://github.com/vortezwohl/JevNER)",
    },
    body: JSON.stringify({ state, model: MODEL, questions }),
  });
  if (!response.ok) {
    throw new Error(`systemone ${response.status}: ${await response.text()}`);
  }
  return (await response.json()) as SystemOneResponse;
}

const TEXT = "Maria Curie visited Paris and met Doctor Strange.";

/** Whitespace tokens keep this demo dependency-free; the SDK ships a real tokenizer. */
const words = TEXT.split(/\s+/).map((w) => w.replace(/[^_\p{L}\p{N}]/gu, ""));

// -- stage 1: detection, one Noul per word, all in a single request ----------
const detect = await systemOne(
  { TEXT },
  Object.fromEntries(
    words.map((word, i) => [
      `w${i}`,
      {
        type: "noul",
        instructions: { INPUT: { NAME: word }, TASK: "Is this word part of a named entity in the text?" },
        criteria: { true: "part of a named entity", false: "not an entity" },
      } satisfies Question,
    ]),
  ),
);

const hits = words
  .map((word, i) => ({ word, noul: (detect.answers[`w${i}`] as NoulAnswer).noul }))
  .filter((h) => h.noul >= 0.5);

// -- stage 2: typing, one Choice per detected word, one more request --------
const typing = hits.length
  ? await systemOne(
      { TEXT },
      Object.fromEntries(
        hits.map((h, i) => [
          `s${i}`,
          {
            type: "choice",
            instructions: { INPUT: { NAME: h.word }, TASK: "Which entity type fits this span best?" },
            criteria: { PERSON: "a named person", GPE: "a city, state, or country", NONE: "none of the above" },
          } satisfies Question,
        ]),
      ),
    )
  : null;

// -- report ------------------------------------------------------------------
console.log(`text  : ${TEXT}`);
console.log(`model : ${detect.model}`);
for (const [i, hit] of hits.entries()) {
  const answer = typing?.answers[`s${i}`] as ChoiceAnswer | undefined;
  const type = answer ? `${answer.choice} @ ${answer.confidence.toFixed(2)}` : "?";
  console.log(`${hit.word.padEnd(16)} noul=${hit.noul.toFixed(2)}  ${type}`);
}
