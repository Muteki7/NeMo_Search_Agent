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

## Future

This state may later get tiers, where the tier sets how much help Claude gives. For now there is one simple state: keyword present means guide only, keyword absent means act normally.
