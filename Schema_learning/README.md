# Schema first, detail second

These four images are the schema for `Debug_checks.md` (the Step 3 index build).
Look at them in order. Then run the debugger and fill in the detail yourself.

| Order | Image | The question it answers | Time |
|---|---|---|---|
| 1 | `01_big_picture.png` | What are the five phases, and where do my 11 stops sit? | 2 min |
| 2 | `02_two_paths.png` | What is the one idea that matters: documents and questions take different paths through the same model? | 3 min |
| 3 | `03_call_map.png` | Who calls whom? This is the shape of the Call Stack panel. | 3 min |
| 4 | `04_stop_map.png` | At each stop: where, what to watch, what to expect? | Keep it open while you debug |

## How to use them

1. Look at image 1. Say the five phases aloud: prepare, build the embedder, embed and store, reopen, test.
2. Look at images 2 and 3. Find the orange "only difference" box in image 2. Find `get_embedding` in image 3.
3. Open `04_stop_map.png` next to the debugger. Start the run from `Debug_checks.md` section 5.
4. At each stop, find the stop's number in image 1. Then see what the debugger shows.
5. After the run, write down every value that differs from the image. Give your notes to Claude.

The details (every step, every value) are in `Debug_checks.md`. The images give the structure that the details fit into.

## Make the images again

Run this from the repository root:

```
NeMo_Gym/resources_servers/mission_search/.venv/bin/python Schema_learning/build_schema.py
```

The script is `build_schema.py`. It needs `matplotlib`, which is in the server venv.
