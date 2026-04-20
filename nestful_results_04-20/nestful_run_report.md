# Nestful Run Report

## Non-PTC

### Overview

| Metric | Value |
| --- | --- |
| Traces | 1861 |
| Scored traces | 1861 |
| Unscored traces | 0 |

### Token Usage

| Metric | Value |
| --- | --- |
| Total token usages | 3711756 |
| Input tokens | 1455984 |
| Output tokens | 62428 |
| Thinking tokens | 2193344 |
| Average total token usages | 1994.495 |
| Average input tokens | 782.366 |
| Average output tokens | 33.545 |
| Average thinking tokens | 1178.584 |
| Average planned tool calls | 1.071 |
| Average token usages for correct win rate | 1973.281 |
| Average token usages for correct full match | N/A |
| Average token usages for tasks both PTC and non-PTC solve | 1929.321 |
| Shared failed tasks (win rate = 0) | 390 |
| Average token usages for tasks both PTC and non-PTC fail on | 2192.359 |
| Average planned tool calls for tasks both PTC and non-PTC fail on | 1.069 |

### Latency

| Metric | Value |
| --- | --- |
| Average latency | 15.406 |
| Average latency for correct win rate | 11.276 |
| Average latency for full match | N/A |

### Quality Metrics

| Metric | Value |
| --- | --- |
| Runtime Errors | 4 |
| Parsing Error | 0 |
| Partial Match Accuracy | 0.168 |
| Full Match Accuracy | 0.000 |
| Win Rate | 0.034 |

### Error Analysis

| Metric | Value |
| --- | --- |
| wrong_first_tool | 723 |
| wrong_first_arguments | 754 |
| starts_correct_then_wrong_tool | 1 |
| starts_correct_then_wrong_arguments | 43 |
| stopped_early | 340 |
| extra_calls | 0 |
| parsing_error | 0 |
| pred_contains_unexpected_tool | 130 |
| pred_contains_unexpected_argument | 137 |

## PTC

### Overview

| Metric | Value |
| --- | --- |
| Traces | 1861 |
| Scored traces | 1861 |
| Unscored traces | 0 |

### Token Usage

| Metric | Value |
| --- | --- |
| Total token usages | 4993948 |
| Input tokens | 2755332 |
| Output tokens | 228056 |
| Thinking tokens | 2010560 |
| Average total token usages | 2683.476 |
| Average input tokens | 1480.565 |
| Average output tokens | 122.545 |
| Average thinking tokens | 1080.365 |
| Average planned tool calls | 3.459 |
| Average token usages for correct win rate | 2564.994 |
| Average token usages for correct full match | 2734.479 |
| Average token usages for tasks both PTC and non-PTC solve | 2952.038 |
| Shared failed tasks (win rate = 0) | 390 |
| Average token usages for tasks both PTC and non-PTC fail on | 3114.828 |
| Average planned tool calls for tasks both PTC and non-PTC fail on | 3.859 |

### Latency

| Metric | Value |
| --- | --- |
| Average latency | 19.531 |
| Average latency for correct win rate | 18.068 |
| Average latency for full match | 15.355 |

### Quality Metrics

| Metric | Value |
| --- | --- |
| Runtime Errors | 38 |
| Parsing Error | 0 |
| Partial Match Accuracy | 0.380 |
| Full Match Accuracy | 0.271 |
| Win Rate | 0.785 |

### Error Analysis

| Metric | Value |
| --- | --- |
| wrong_first_tool | 680 |
| wrong_first_arguments | 353 |
| starts_correct_then_wrong_tool | 141 |
| starts_correct_then_wrong_arguments | 111 |
| stopped_early | 67 |
| extra_calls | 4 |
| parsing_error | 0 |
| pred_contains_unexpected_tool | 297 |
| pred_contains_unexpected_argument | 306 |
