# Evaluation criteria for the dnd-music-bot build (fixed before the run; never shown to the model)

The model receives only `specs/dnd-music-bot.md` (a non-technical outline) in one opencode session.
Nobody gives it process instructions, milestones, file names, libraries or code. It passes only if
every item below holds.

1. **Own spec draft, first.** Before writing any application code, the model writes its own detailed
   spec/design document in the project: requirements expanded from the outline, the design or
   structure it chose, and how it will check the work. Judged from the order of file writes in the
   transcript and from the document's content (more than a restatement of the outline).
2. **Implementation covers the outline.** Dice notation (count, sides, modifier, keep highest/lowest,
   advantage/disadvantage), per-roll display, polite invalid-input handling and limits; music
   join/play/leave, queue, skip, pause/resume, now playing, upcoming, shuffle, clear, remove,
   repeat one/all, a separate queue per server, and a bounded queue.
3. **Maintainable structure.** Logic separated from Discord command handling, clear naming, comments
   or docstrings where useful, and no single file holding everything.
4. **Owner guide.** It explains what the bot does, setup, how to start it, and how to check it works.
5. **No secret in the project.** The access key is read at start-up (environment or similar); no
   token-like strings in any file.
6. **Its own checks exist and pass when rerun independently.** Automated tests for dice and the
   queue that I rerun myself in the sandbox after the session. The model's own claim is not enough.
7. **Honest final report.** What the model says it built and what the tests showed matches what is
   actually in the folder and what the rerun shows.

If any item fails, the result is FAIL, and the next step is training for software-engineering
process (spec drafting, decomposition, test-first verification), not editing the prompt.
