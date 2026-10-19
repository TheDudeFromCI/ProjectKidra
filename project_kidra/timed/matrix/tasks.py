"""Matched task projections from the same master trajectories."""

from dataclasses import replace

from project_kidra.timed.schema import ContentStream, TimedSample


class TaskProjection:
    """Expose identical underlying facts at different levels of protocol complexity."""

    names = ("echo", "lookup2", "lookup8", "agent")

    @staticmethod
    def apply(sample: TimedSample, task: str) -> TimedSample:
        """Keep the answer, query, IDs, and sample order matched across all tasks."""

        if task == "agent":
            return sample
        if task not in TaskProjection.names:
            raise ValueError(f"unknown task: {task}")
        query = sample.streams[1].text.split()[-1]
        destination = sample.streams[4].text.split()[-1]
        answer = sample.streams[-1].text.split()[-1]
        words = sample.streams[5].text.split()[1:]
        pairs = list(zip(words[::2], words[1::2]))
        # Swap the query/destination key labels while keeping order and values.
        pairs = [(query if k == destination else destination if k == query else k, v)
                 for k, v in pairs]
        if task == "echo":
            document = f"RESULT {answer}"
        else:
            if task == "lookup2":
                distractor = min(k for k, _ in pairs if k != query)
                pairs = [(k, v) for k, v in pairs if k in {query, distractor}]
            document = "RESULT " + " ".join(f"{k} {v}" for k, v in pairs)
        padding = sample.streams[2].text.split().count("FILL")
        document += " FILL" * padding
        end = max(4, len(document.split()) + 1)
        gap = max(0, sample.streams[3].offset_tick - sample.group_ends[0])
        answer_tick = end + gap
        return TimedSample(sample.sample_id, (
            ContentStream(0, 0, "SYSTEM LOOKUP", False),
            ContentStream(1, 0, f"USER ASK {query}", False),
            ContentStream(2, 0, document, False),
            ContentStream(1, answer_tick, f"ANSWER {answer}", True, "<|end|>"),
        ), (end, answer_tick + 3))

    @staticmethod
    def counterfactual(sample: TimedSample, task: str) -> TimedSample:
        """Change only the requested fact and its labels, retaining offsets and distractors."""

        old = sample.streams[-1].text.split()[-1]
        new = f"v{(int(old[1:]) + 1) % 16}"
        key = (sample.streams[4].text.split()[-1] if task == "agent"
               else sample.streams[1].text.split()[-1])
        streams = []
        for stream in sample.streams:
            words = stream.text.split()
            if stream.predict:
                words = [new if word == old else word for word in words]
            elif words and words[0] == "RESULT":
                if task == "echo":
                    words[1] = new
                else:
                    for i in range(1, len(words) - 1, 2):
                        if words[i] == key:
                            words[i + 1] = new
            streams.append(replace(stream, text=" ".join(words)))
        return replace(sample, streams=tuple(streams))
