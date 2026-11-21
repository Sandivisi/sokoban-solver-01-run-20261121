[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Sandivisi/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Sandivisi/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

# Sokoban solver

A push-optimal Sokoban solver built for make.fun's
[Build the best Sokoban solver, 21 November 2026](https://make.fun/bounty/sokoban-solver-01-run-20261121)
bounty.

## How it works

Every level the competition draws is a hallway with one to four rooms off
it, each room holding 3 or 4 boxes and as many goals, and a box can never
leave its room — a push through a door strands it in the hallway, where it
reaches no goal. So the level splits into independent rooms, and the
level's push optimum is the sum of the rooms' optima. Each room is solved
with an A* search over **pushes** (the player only walks between pushes),
which is optimal because its heuristic is admissible:

- **Heuristic:** the cheapest assignment of boxes to goals, where the cost
  of one box to one goal is the shortest walk over open floor (ignoring
  other boxes). One push moves one box one square, so this sum can drop by
  at most one per push.
- **Pruning:** a box is only ever pushed onto a *live* square (one from
  which some goal is reachable by pushes, ignoring other boxes), and a
  push that leaves a box frozen inside a 2x2 block of walls and boxes is
  discarded.
- **States:** the boxes' squares plus the player's reachable region
  (named by its lowest square), so walking never creates new states.

The search is bounded by a fixed number of expanded states, never by the
clock, and every tie is broken by insertion order, so two runs write the
same bytes. No network, no randomness, no wall clock. If a level does not
split into rooms, the same push search runs over the whole level as a
fallback.

## Run

```sh
node main.ts
```

Reads each instance from `/input/<id>/level.txt` (or
`$MAKEFUN_INPUT_DIR`) and writes its answer to
`/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`): the player's
moves in LURD (`u`, `d`, `l`, `r`), one letter a step.

Requires Node 22 or newer. No dependencies, nothing to install.

## License

MIT — see [LICENSE](LICENSE).
