# Ablation results

TCE = % of (exam, rubric code) pairs whose deduction differs between grading the expert transcript and grading this transcript (lower is better). Claim precision = helpful / (helpful + harmful) accepted claims, judged against the expert transcript. Accepted / Escalated / Rejected are Arbiter decisions over all exams. Unless stated otherwise: fine-tuned reader, disputed value masked (`ft_mask`), full evidence.

## Transcriber: `qwen3b_ft`
Exams: 80 · claims ruled on (interactive): 112 · θ = 0.85

### #9 Headline: system comparison

| System / variant | n | TCE % | 95% CI | Wrongful % | Over-credit % | Exams w/ number error % | Mean number mismatches | Accepted | Escalated | Rejected | Claim precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No correction (`none`) | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | - | - | - |
| B2 one-pass self-correction | 80 | 25.0 | 20.8–29.4 | 18.5 | 6.5 | 95.0 | 14.7 | - | - | - | - |
| One-shot agents + Arbiter | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | 45 | 26 | - |
| Interactive agents + Arbiter (verify pass) | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | 104 | 8 | - |
| Interactive, as built (agent-requested evidence only) | 55 | 26.9 | 20.0–33.5 | 23.3 | 3.6 | 90.9 | 10.2 | - | - | - | - |

### #2 Evidence used by the Arbiter (interactive run)

| System / variant | n | TCE % | 95% CI | Wrongful % | Over-credit % | Exams w/ number error % | Mean number mismatches | Accepted | Escalated | Rejected | Claim precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Re-read + consistency (full) | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | 104 | 8 | - |
| Re-read only | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | 104 | 8 | - |
| Consistency only | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | 2 | 106 | 4 | 0.0 |
| No verification (accept every claim) | 80 | 28.5 | 23.6–33.5 | 23.2 | 5.2 | 96.2 | 12.8 | 40 | - | - | 8.7 |

### #3 Threshold sweep (interactive run, full evidence)

| θ | Accepted | Escalated | Rejected | Accepted helpful | Accepted harmful | Claim precision % | TCE % (if scored) | Exams w/ number error % (if scored) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 2 | 99 | 11 | 2 | - | 100.0 | 28.2 | 92.5 |
| 0.6 | 2 | 99 | 11 | 2 | - | 100.0 | - | - |
| 0.7 | - | 101 | 11 | - | - | - | 28.2 | 92.5 |
| 0.8 | - | 104 | 8 | - | - | - | - | - |
| 0.85 | - | 104 | 8 | - | - | - | 28.2 | 92.5 |
| 0.9 | - | 104 | 8 | - | - | - | - | - |
| 0.95 | - | 109 | 3 | - | - | - | 28.2 | 92.5 |

### #6 Base vs fine-tuned reader · #7 Disputed value hidden vs shown (interactive run)

Verifier accuracy = of the re-reads that picked either the transcribed or the proposed value, how often the pick agrees with the expert transcript.

| Reader config | Verifier accuracy % | Evaluable | Picked proposed | Picked transcribed | Unresolved | No reading | Mean conf | Accepted | Claim precision % | TCE % | Exams w/ number error % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ft_mask | 100.0 | 12 | 2 | 15 | 10 | 85 | 0.8 | - | - | 28.2 | 92.5 |
| base_mask | 100.0 | 14 | 4 | 17 | 7 | 84 | 0.9 | - | - | 28.2 | 92.5 |
| ft_nomask | 75.0 | 16 | 2 | 24 | 1 | 85 | 0.9 | - | - | 28.2 | 92.5 |
| base_nomask | 89.5 | 19 | - | 29 | - | 83 | 1.0 | - | - | 28.2 | 92.5 |

### #10 True single-agent control

Post-hoc = that agent's claims taken out of the joint (interactive) run; solo = the agent ran alone.

| System / variant | n | TCE % | 95% CI | Wrongful % | Over-credit % | Exams w/ number error % | Mean number mismatches | Accepted | Escalated | Rejected | Claim precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Advocate, post-hoc from joint run | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | - | - | - |
| Advocate, solo run | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | 49 | 8 | - |
| Auditor, post-hoc from joint run | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | - | - | - |
| Auditor, solo run | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | 36 | 2 | - |
| Both agents (interactive) | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | 104 | 8 | - |

### Interaction and agent-tool stats

| Run | Filed r1 | Filed r2 | Countered | Withdrawn | Revised | Conflicts after r3 | Rejected actions | crop_reread calls | Tool ok | Tool failed |
|---|---|---|---|---|---|---|---|---|---|---|
| interactive | 99 | 31 | 23 | 15 | 2 | 22 | 109 | 183 | 46 | 157 |
| solo_advocate | 53 | 8 | - | 3 | 1 | 2 | 56 | 94 | 23 | 74 |
| solo_auditor | 46 | 2 | - | 9 | 1 | 0 | 11 | 55 | 18 | 47 |

## Transcriber: `gemini25`
Exams: 80 · claims ruled on (interactive): 123 · θ = 0.85

### #9 Headline: system comparison

| System / variant | n | TCE % | 95% CI | Wrongful % | Over-credit % | Exams w/ number error % | Mean number mismatches | Accepted | Escalated | Rejected | Claim precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No correction (`none`) | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | - | - | - | - |
| B2 one-pass self-correction | 80 | 7.5 | 4.8–10.2 | 5.1 | 2.4 | 71.2 | 6.9 | - | - | - | - |
| One-shot agents + Arbiter | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | - | 62 | 5 | - |
| Interactive agents + Arbiter (verify pass) | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | 2 | 114 | 7 | 0.0 |
| Interactive, as built (agent-requested evidence only) | 67 | 9.2 | 4.2–15.5 | 6.0 | 3.3 | 49.2 | 1.3 | - | - | - | - |

### #2 Evidence used by the Arbiter (interactive run)

| System / variant | n | TCE % | 95% CI | Wrongful % | Over-credit % | Exams w/ number error % | Mean number mismatches | Accepted | Escalated | Rejected | Claim precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Re-read + consistency (full) | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | 2 | 114 | 7 | 0.0 |
| Re-read only | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | 2 | 114 | 7 | 0.0 |
| Consistency only | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 51.2 | 1.4 | 3 | 109 | 11 | 0.0 |
| No verification (accept every claim) | 80 | 8.0 | 4.2–12.2 | 5.0 | 3.0 | 68.8 | 2.1 | 51 | - | - | 0.0 |

### #3 Threshold sweep (interactive run, full evidence)

| θ | Accepted | Escalated | Rejected | Accepted helpful | Accepted harmful | Claim precision % | TCE % (if scored) | Exams w/ number error % (if scored) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 2 | 111 | 10 | - | 2 | 0.0 | 8.5 | 48.8 |
| 0.6 | 2 | 111 | 10 | - | 2 | 0.0 | - | - |
| 0.7 | 2 | 111 | 10 | - | 2 | 0.0 | 8.5 | 48.8 |
| 0.8 | 2 | 114 | 7 | - | 2 | 0.0 | - | - |
| 0.85 | 2 | 114 | 7 | - | 2 | 0.0 | 8.5 | 48.8 |
| 0.9 | 2 | 117 | 4 | - | 2 | 0.0 | - | - |
| 0.95 | - | 123 | - | - | - | - | 8.5 | 48.8 |

### #6 Base vs fine-tuned reader · #7 Disputed value hidden vs shown (interactive run)

Verifier accuracy = of the re-reads that picked either the transcribed or the proposed value, how often the pick agrees with the expert transcript.

| Reader config | Verifier accuracy % | Evaluable | Picked proposed | Picked transcribed | Unresolved | No reading | Mean conf | Accepted | Claim precision % | TCE % | Exams w/ number error % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ft_mask | 72.7 | 22 | 6 | 17 | 11 | 89 | 0.9 | 2 | 0.0 | 8.5 | 48.8 |
| base_mask | 90.9 | 22 | 2 | 21 | 11 | 89 | 0.9 | - | - | 8.5 | 48.8 |
| ft_nomask | 96.7 | 30 | 1 | 31 | 2 | 89 | 0.9 | 1 | 0.0 | 8.5 | 48.8 |
| base_nomask | 100.0 | 32 | - | 34 | - | 89 | 1.0 | - | - | 8.5 | 48.8 |

### #10 True single-agent control

Post-hoc = that agent's claims taken out of the joint (interactive) run; solo = the agent ran alone.

| System / variant | n | TCE % | 95% CI | Wrongful % | Over-credit % | Exams w/ number error % | Mean number mismatches | Accepted | Escalated | Rejected | Claim precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Advocate, post-hoc from joint run | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | - | - | - | - |
| Advocate, solo run | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | 1 | 66 | 6 | 0.0 |
| Auditor, post-hoc from joint run | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | - | - | - | - |
| Auditor, solo run | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | 2 | 36 | 2 | 0.0 |
| Both agents (interactive) | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | 2 | 114 | 7 | 0.0 |

### Interaction and agent-tool stats

| Run | Filed r1 | Filed r2 | Countered | Withdrawn | Revised | Conflicts after r3 | Rejected actions | crop_reread calls | Tool ok | Tool failed |
|---|---|---|---|---|---|---|---|---|---|---|
| interactive | 116 | 32 | 24 | 21 | 4 | 16 | 125 | 207 | 73 | 155 |
| solo_advocate | 65 | 12 | - | 2 | 2 | 3 | 74 | 114 | 34 | 83 |
| solo_auditor | 51 | 7 | - | 12 | 6 | 0 | 15 | 60 | 20 | 48 |

## #16 Transcript quality

Compare each section above across transcribers. Caveat: the expert transcripts were made by correcting Gemini 2.5 Pro's output, which favours Gemini.

| Transcriber | System / variant | n | TCE % | 95% CI | Wrongful % | Over-credit % | Exams w/ number error % | Mean number mismatches | Accepted | Escalated | Rejected | Claim precision % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b_ft | none | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | - | - | - |
| qwen3b_ft | interactive + Arbiter | 80 | 28.2 | 23.0–33.5 | 23.0 | 5.2 | 92.5 | 12.5 | - | 104 | 8 | - |
| gemini25 | none | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | - | - | - | - |
| gemini25 | interactive + Arbiter | 80 | 8.5 | 4.4–13.7 | 5.5 | 3.0 | 48.8 | 1.4 | 2 | 114 | 7 | 0.0 |
