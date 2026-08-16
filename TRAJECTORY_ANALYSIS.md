# What the winning trajectories do differently

Analysis of all 48 public trajectories in `txbench-pp/trajectories/` — 12 evals ×
4 model/harness pairs.

**Method.** The trajectories record no score, so I reconstructed each run's final
answer by scanning assistant text, tool arguments, and tool output for a JSON
object matching the eval's expected answer fields, then graded it with the same
`latch-eval-tools` graders the benchmark uses. Caveat: one run per pair on 12
evals, so treat individual numbers as directional. The resulting ranking matches
the published 100-eval leaderboard ordering, which is a decent sanity check.

## Scoreboard

| Model | Harness | Passed | Wrong answer | **No answer at all** |
|---|---|---:|---:|---:|
| Claude Opus 4.8 | claude-code | **6/12** | 5 | 1 |
| GPT-5.5 | openai-codex | 5/12 | 7 | 0 |
| Gemini 3.5 Flash | pi | 3/12 | 4 | 5 |
| Grok 4.3 | pi | 0/12 | 5 | 7 |

## Finding 1: the top failure mode is not answering

**13 of 48 runs (27%) never produced a final answer.** They ran out of turns or
stalled mid-reasoning — Gemini's `PRISM_V10_C` trajectory ends on "I'm now writing
a Python script to ensure that either 863 or 948 can be accurately generated" and
simply stops. Grok's `H2_toxfree` run ends mid-paragraph of prose.

This is where the field separates. Grok and Gemini forfeited 12 evals between them
without being wrong about anything. Opus forfeited 1, GPT-5.5 zero.

## Finding 2: failures are judgment, not computation

`CTRL01` is the clearest case. Ground truth is `{responsive: 16, removed: 0}` —
the task deliberately tempts you to gate the crizotinib hit list on a second
control compound's readout, and the correct move is to refuse.

| Model | Answer | |
|---|---|---|
| Opus 4.8 | `{16, 0}` | pass |
| GPT-5.5 | `{16, 5}` | fail |
| Gemini 3.5 | `{16, 5}` | fail |
| Grok 4.3 | `{16, 5}` | fail |

All four parsed the CSV correctly and all four computed 16. Three then applied a
filter nobody asked for. The benchmark is not testing data wrangling; it is
testing whether the agent over-processes.

Four evals went 0/4: `HAI2027_MECH01`, `PRISM_V10_C`, `lee2024_RNA1`, and
`sp_01_plate_well_position_confounder` — on the last, three of four produced no
answer at all.

## Finding 3: tool usage

Every harness is bash-dominated. Nothing exotic appears anywhere.

| Model | Tool calls (all 12 runs) |
|---|---|
| Opus 4.8 | `Bash` 184, `Read` 29, `Write` 14, `TaskCreate/Update` 8, `Edit` 2, `ToolSearch` 1 |
| GPT-5.5 | `command_execution` 180, `file_change` 8, `web_search` 4, `todo_list` 1 |
| Gemini 3.5 | `bash` 376, `write` 60, `read` 20 |
| Grok 4.3 | `bash` 78, `write` 7, `read` 5 |

`web_search` appears 4 times across all 48 runs, all GPT-5.5. Nobody needs
external knowledge — the answer is always in the staged data.

**Volume is not the differentiator; ratio is.** Median text generated per tool
call: Opus 3,500 chars, GPT-5.5 4,800, Grok 10,300, Gemini 10,500. The two losers
generate two to three times more reasoning per action. Grok's median run is 7 tool
calls against 72K characters of deliberation — it thinks instead of acting.
Gemini's is the opposite pathology: 31 tool calls, 419K characters, thrashing.

Opus does the least talking per unit of work.

## Finding 4: winners write scripts, losers write one-liners

Averaged per trajectory:

| | Passed | Failed (answered) | No answer |
|---|---:|---:|---:|
| `.py` files written | **6.0** | 0.7 | 4.4 |
| heredoc commands | 4.4 | 4.0 | 0.9 |
| median command length | 446 | 362 | 235 |
| inspection cmds (`ls`/`head`/`wc`) | 4.9 | 5.6 | 7.2 |

Passing runs commit their analysis to a file — nearly 9× more often than runs that
answered but got it wrong. Failing runs chain `python3 -c "..."` one-liners,
losing state between calls and re-deriving intermediates.

Note the last row: **inspection is inversely correlated with success.** Runs that
never answered poked at the data the most. Exploration is not the bottleneck.

## What this implies for our agent

1. **Always emit an answer.** A guess scores no worse than silence, and silence
   cost the bottom two models 12 evals. Write `eval_answer.json` early with a
   best guess and overwrite as confidence improves.
2. **Budget the deliberation, not the tool calls.** The winning profile is ~19
   tool calls with low reasoning-per-action. Force action early.
3. **Write a script file, don't chain one-liners.** Strongest style signal in
   the data.
4. **Prompt explicitly against over-filtering.** The single most common wrong
   answer is applying a gate the task never asked for. Something like: *apply
   exactly the criteria stated in the task; do not add filters, quality gates, or
   exclusions the task did not specify, even where they seem scientifically
   prudent.*
5. **Skip web search and exotic tooling.** Bash, Python, and pandas cover it.
