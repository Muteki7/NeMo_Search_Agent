# Rules

Constraints on how Claude performs implementations in this project. Each rule is a named state. A rule applies only when its keyword appears in the chat.

## DELIBERATE_PRACTICE

The keyword is spelled exactly `DELIBERATE_PRACTICE` Match it exactly.

**Purpose:** the user wants to write the code changes themselves, to stay in the loop and get more involved in the implementation.

**When the keyword appears in the chat:**
- Do not make the changes yourself. Do not edit or write project files, and do not install or change packages.
- Tell the user what to change and where to look to change it: the file, the function or line area, what to add, remove or replace, and why.
- Present each change as a step the user can do, in a sensible order, so they can follow along.
- Reading files and running read-only commands is fine, so the guidance points at the right place.

**When the keyword does not appear anywhere in the chat:** act normally. Make the changes and implementations yourself, the way you normally would.

## DELIBERATE_PRACTICE_v0

The keyword is spelled exactly `DELIBERATE_PRACTICE_v0`. Match it exactly. Because it contains `DELIBERATE_PRACTICE`, it takes precedence: if `DELIBERATE_PRACTICE_v0` appears, follow this section.

**Purpose:** same as `DELIBERATE_PRACTICE` (the user writes the code changes themselves), but with a worked example of each change.

**When the keyword appears in the chat:**
- Everything from `DELIBERATE_PRACTICE` still applies: do not edit or write project files, do not install or change packages, give clear step-by-step guidance on what to change and where, and read-only commands are fine.
- In addition, for each change, give a small code chunk that is representative of the change. It should show the shape of what to write, so the user can adapt it and place it themselves.
- Keep each chunk short and tied to a specific step, with its location (file and line area) stated next to it.

**Tier summary:** `DELIBERATE_PRACTICE` = steps only. `DELIBERATE_PRACTICE_v0` = steps plus representative code chunks. No keyword = Claude makes the changes.

## Future

This state may later get tiers, where the tier sets how much help Claude gives. For now there is one simple state: keyword present means guide only, keyword absent means act normally.

## DELIBERATE_DEBUG

The keyword is spelled exactly `DELIBERATE_DEBUG`. Match it exactly.

**Purpose:** the user wants to understand a code file in depth by stepping through it with the Python debugger in VS Code. After the user does everything in the guidance, they should clearly understand the file and its important operations.

**When the keyword appears in the chat:**
- Do not edit or write project files, and do not install or change packages. Do not add print statements or other changes to make debugging easier. Reading files is fine, and so are read-only commands.
- Read the file first, and the files it calls when needed. Read it again right before you give line numbers, because lines move and the user may have edited it.
- Give a numbered walkthrough that the user can follow in order:
  1. **Setup:** which file or function to run, the arguments, the working directory, the Python interpreter or venv to select, and any environment variables. If the run needs a launch configuration, describe what it must contain. Do not create the file.
  2. **Breakpoints:** each with the file and line (or the function name, when line numbers may move) and the reason that place matters. Add a condition, a hit count or a logpoint when it helps, for example to skip the first loop iterations.
  3. **Moving through the code:** at each stop, say whether to Continue (F5), Step Over (F10), Step Into (F11), Step Out (Shift+F11) or Run to Cursor. Say when to step into a call and when to step over it, and why. Step into the project's own code. Step over library code, unless the point is to learn what the library does.
  4. **What to look at:** the variables in the Locals and Globals panels, with the types, shapes or values to expect and what should change after each step. Which frames in the Call Stack to look at, and what the stack should look like. Exact expressions to add to Watch or to run in the Debug Console. Say when a Debug Console expression can change state.
  5. **What to learn at each stop:** one or two sentences on what the operation does and why, and one thing for the user to confirm in the debugger.
- Order the stops along the real flow of the program, from start to end. Cover the important operations, not every line. Use only as many breakpoints as needed, usually 5 to 12.
- Say when an expected value comes from a real run and when it is an estimate from reading the code.
- Warn when a breakpoint cannot hit, for example code that runs in another process, a compiled library, a generated file, or the GPU. Name the nearest place where a breakpoint does work.
- Run the code only when the run has no side effects: it writes no files, calls no outside service, and does not use the GPU while another job needs it. If unsure, ask first.
- End with a short recap of the flow in the order the user saw it, and 2 to 4 questions that the user can answer from what they saw in the debugger.

**Other keywords:** this rule is independent of `DELIBERATE_PRACTICE` and `DELIBERATE_PRACTICE_v0`. If one of them also appears, follow both rules.

**When the keyword does not appear anywhere in the chat:** act normally. If the user asks about a code file without the keyword, answer the way you normally would.
