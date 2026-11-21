// Sokoban solver: push-optimal A* over pushes, solved room by room.
//
// Strategy
// --------
// Every level this competition draws is a hallway with one to four rooms
// off it, each room holding 3 or 4 boxes and as many goals, and a box can
// never leave its room (a push through a door strands it in the hallway,
// where it reaches no goal). So the level splits into independent rooms
// and the level's push optimum is the sum of the rooms' optima. Each room
// is solved with an A* search over *pushes* (the player only walks between
// pushes), which is optimal when the heuristic is admissible:
//
//   - heuristic: the cheapest assignment of boxes to goals, where the cost
//     of one box to one goal is the shortest walk over open floor (ignoring
//     other boxes). One push moves one box one square, so this sum can drop
//     by at most one per push: admissible and consistent.
//   - pruning: a box is only ever pushed onto a "live" square (a square
//     from which some goal is reachable by pushes, ignoring other boxes),
//     and a push that leaves a box frozen inside a 2x2 block of walls and
//     boxes is discarded.
//   - a state is the boxes' squares plus the player's reachable region
//     (named by its lowest square), so walking never creates new states.
//
// The search is bounded by a fixed number of expanded states, never by
// the clock, and every tie is broken by insertion order, so two runs write
// the same bytes. No network, no randomness, no wall clock.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// How many states one room search may expand, and how many states the
// whole run may expand over all its searches. Work limits, not time.
const ROOM_STATE_CAP = 400_000;
const WHOLE_STATE_CAP = 1_500_000;
const GLOBAL_STATE_CAP = 4_000_000;

// The four steps, in LURD order: [dx, dy, letter]. The delta in the
// board's flat index is dy * w + dx and is filled in per level.
const DIRS: { dx: number; dy: number; ch: string }[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

type Level = {
  w: number;
  h: number;
  n: number;
  wall: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player, + player on goal). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x] as string;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goal[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, n, wall, goal, goals, boxes, player };
}

/**
 * The live squares: squares from which a box, alone on the board, can be
 * pushed onto some goal. Found by pulling boxes back from the goals: a box
 * on square y can be pulled from a live square x = y + d exactly when the
 * player can stand at y - d (open floor) to push it to x.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, n, wall, goals } = level;
  const live = new Uint8Array(n);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi] as number;
    for (const { dx, dy } of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= n || live[y] || wall[y] || wall[y - d]) continue;
      live[y] = 1;
      queue.push(y);
    }
  }
  return live;
}

/** Walking distances over open floor from one square, -1 where unreachable. */
function floorDistances(level: Level, from: number): Int32Array {
  const { w, n, wall } = level;
  const dist = new Int32Array(n).fill(-1);
  const queue: number[] = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi] as number;
    for (const { dx, dy } of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= n || wall[q] || dist[q] >= 0) continue;
      dist[q] = dist[p] as number + 1;
      queue.push(q);
    }
  }
  return dist;
}

/**
 * The cheapest one-to-one assignment of boxes to goals, summing each
 * box's floor distance to its goal (Infinity when some box reaches no
 * goal). Enumerating the permutations is cheap for 3 or 4 boxes.
 */
function matchingHeuristic(boxes: number[], goalDists: Int32Array[]): number {
  const k = boxes.length;
  const order: number[] = [];
  for (let i = 0; i < k; i++) order.push(i);
  let best = Infinity;
  const permute = (at: number, cost: number): void => {
    if (cost >= best) return;
    if (at === k) {
      best = cost;
      return;
    }
    for (let j = at; j < k; j++) {
      const g = order[j] as number;
      const d = goalDists[g]?.[boxes[at] as number] as number;
      if (d < 0) continue;
      const old = order[at] as number;
      order[at] = g;
      order[j] = old;
      permute(at + 1, cost + d);
      order[j] = g;
      order[at] = old;
    }
  };
  permute(0, 0);
  return best;
}

/**
 * Flood fill from `from` over floor squares not holding a box. `seen`
 * marks squares with `stamp`; `stack` is scratch. Returns the lowest
 * square reached, which names the player's region.
 */
function floodLow(
  level: Level,
  occ: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, n, wall } = level;
  let low = from;
  let sp = 0;
  stack[sp++] = from;
  seen[from] = stamp;
  while (sp > 0) {
    const p = stack[--sp] as number;
    if (p < low) low = p;
    for (const { dx, dy } of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= n || wall[q] || occ[q] || seen[q] === stamp) continue;
      seen[q] = stamp;
      stack[sp++] = q;
    }
  }
  return low;
}

/** True when the box just pushed onto `t` sits frozen inside a 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(level: Level, occ: Uint8Array, t: number): boolean {
  const { w, wall, goal } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

type SearchResult = { pushes: number; moves: string; player: number };

/** A state's key: its region's lowest square, then its boxes, packed base-n. Exact while n ** (boxes + 1) < 2 ** 53. */
function stateKey(boxes: number[], low: number, n: number): number | string {
  if (boxes.length <= 4 && Math.pow(n, boxes.length + 1) < 9007199254740992) {
    let k = low;
    for (const b of boxes) k = k * n + b;
    return k;
  }
  return `${low}|${boxes.join(',')}`;
}

// Shared across a run so the total work stays bounded by count, never by the clock.
let globalExpansions = 0;

/**
 * The push-optimal solution of `level` (or null when the search gives up):
 * A* over pushes. A state is the sorted boxes plus the player's region;
 * the goal test runs when a state leaves the heap, and the heuristic is
 * the cheapest box-to-goal assignment over floor distances, so the first
 * solved state popped is push-optimal.
 */
function pushSearch(level: Level, maxStates: number): SearchResult | null {
  const { w, n, wall, goal, goals } = level;
  const live = liveSquares(level);
  const goalDists = goals.map((g) => floorDistances(level, g));
  const occ = new Uint8Array(n);
  const seen = new Int32Array(n);
  const mark = new Int32Array(n);
  const stack = new Int32Array(n);
  let stamp = 0;
  let markStamp = 0;

  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);

  // The states in discovery order: boxes, player square, parent, push from/to.
  const sBoxes: number[][] = [level.boxes.slice()];
  const sPlayer: number[] = [level.player];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sG: number[] = [0];

  const startLow = (() => {
    for (const b of level.boxes) occ[b] = 1;
    const low = floodLow(level, occ, level.player, seen, ++stamp, stack);
    for (const b of level.boxes) occ[b] = 0;
    return low;
  })();
  const best = new Map<number | string, number>();
  best.set(stateKey(level.boxes, startLow, n), 0);

  // The open list: a binary heap of [f, g, tie, state], smallest f first,
  // deeper g first on a tie, then earliest inserted.
  const heap: [number, number, number, number][] = [];
  let tie = 0;
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i] as [number, number, number, number], heap[p] as [number, number, number, number])) break;
      const tmp = heap[i] as [number, number, number, number];
      heap[i] = heap[p] as [number, number, number, number];
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0] as [number, number, number, number];
    const last = heap.pop() as [number, number, number, number];
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1] as [number, number, number, number], heap[c] as [number, number, number, number])) c++;
        if (!heapLess(heap[c] as [number, number, number, number], heap[i] as [number, number, number, number])) break;
        const tmp = heap[i] as [number, number, number, number];
        heap[i] = heap[c] as [number, number, number, number];
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  const h0 = matchingHeuristic(level.boxes, goalDists);
  if (!Number.isFinite(h0)) return null;
  heapPush([h0, 0, tie++, 0]);
  if (solved(level.boxes)) return { pushes: 0, moves: '', player: level.player };

  let expanded = 0;
  while (heap.length > 0) {
    const [, g, , s] = heapPop();
    const boxes = sBoxes[s] as number[];
    // A stale heap entry: a cheaper route to this state was found since.
    const low0 = floodLow(level, (() => { for (const b of boxes) occ[b] = 1; return occ; })(), sPlayer[s] as number, seen, ++stamp, stack);
    for (const b of boxes) occ[b] = 0;
    if ((best.get(stateKey(boxes, low0, n)) as number) < g) continue;
    if (solved(boxes)) {
      // Rebuild the moves: walk to each push square, then push.
      const path: number[] = [];
      for (let t = s; t > 0; t = sParent[t] as number) path.push(t);
      path.reverse();
      let moves = '';
      let player = level.player;
      let cur = level.boxes.slice();
      for (const t of path) {
        const from = sFrom[t] as number;
        const to = sTo[t] as number;
        const d = to - from;
        const dir = DIRS.find((dd) => dd.dy * w + dd.dx === d) as { dx: number; dy: number; ch: string };
        for (const b of cur) occ[b] = 1;
        const walk = walkTo(level, occ, player, from - d);
        for (const b of cur) occ[b] = 0;
        if (walk === null) return null;
        moves += walk + dir.ch;
        player = from;
        const next = sBoxes[t] as number[];
        cur = next;
      }
      return { pushes: path.length, moves, player };
    }
    if (expanded >= maxStates || globalExpansions >= GLOBAL_STATE_CAP) return null;
    expanded++;
    globalExpansions++;

    for (const b of boxes) occ[b] = 1;
    const reachStamp = ++stamp;
    floodLow(level, occ, sPlayer[s] as number, seen, reachStamp, stack);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi] as number;
      for (const { dx, dy } of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        const behind = b - d;
        if (behind < 0 || behind >= n || seen[behind] !== reachStamp) continue;
        if (t < 0 || t >= n || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        const frozen = frozenBlock(level, occ, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        let h = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          // A separate mark array: this flood must not overwrite the
          // reachability marks the push loop below still reads.
          const low = floodLow(level, occ, b, mark, ++markStamp, stack);
          key = stateKey(next, low, n);
          h = matchingHeuristic(next, goalDists);
        }
        occ[t] = 0;
        occ[b] = 1;
        if (frozen || !Number.isFinite(h)) continue;
        const g2 = g + 1;
        const known = best.get(key);
        if (known !== undefined && known <= g2) continue;
        best.set(key, g2);
        const idx = sBoxes.length;
        sBoxes.push(next as number[]);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sG.push(g2);
        heapPush([g2 + h, g2, tie++, idx]);
      }
    }
    for (const b of boxes) occ[b] = 0;
  }
  return null;
}

/**
 * The shortest walk from `from` to `to` around walls and the boxes in
 * `occ`, in lower LURD letters (u first on ties), or null when unreachable.
 */
function walkTo(level: Level, occ: Uint8Array, from: number, to: number): string | null {
  const { w, n, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(n).fill(-1);
  const how = new Array<string>(n);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi] as number;
    if (p === to) break;
    for (const { dx, dy, ch } of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= n || wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = ch;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p] as number) s = (how[p] as string) + s;
  return s;
}

type Room = { boxes: number[]; goals: number[]; done: boolean };

/**
 * The level's rooms: cut the dead squares of the player's row (the
 * hallway: a box there reaches no goal) out of the floor, and each part
 * left that holds a box is a room. Null when the level is no set of rooms
 * (a box starts in the hallway, a goal is cut off, or some part holds a
 * different count of boxes than goals).
 */
function roomsOf(level: Level): Room[] | null {
  const { w, n, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(n);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(n).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < n; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i] as number;
      for (const { dx, dy } of DIRS) {
        const q = p + dy * w + dx;
        if (q < 0 || q >= n || wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = rooms.length;
        squares.push(q);
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    const r = part[b] as number;
    if (r < 0) return null;
    (rooms[r] as Room).boxes.push(b);
  }
  for (const g of goals) {
    const r = part[g] as number;
    if (r < 0) return null;
    (rooms[r] as Room).goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The square the player ends on after playing `moves` from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * Solve the level room by room: each room's push-optimal solution in
 * turn, the other rooms' boxes standing as walls (on their goals once
 * their room is done), the player walking on from where the last room
 * left it. Null when a room's search gives up.
 */
function solveByRooms(level: Level): string | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      n: level.n,
      wall,
      goal: level.goal,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = pushSearch(sub, ROOM_STATE_CAP);
    if (solved === null) return null;
    moves += solved.moves;
    player = solved.player;
    room.done = true;
  }
  // Sanity: the player must end where the moves say it does.
  if (endOf(level.w, level.player, moves) !== player) return null;
  return moves;
}

/** Solve one level's text: room by room when it splits, else whole. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.boxes.length === 0) return '';
  if (level.boxes.every((b) => level.goal[b])) return '';
  const byRooms = solveByRooms(level);
  if (byRooms !== null) return byRooms;
  const whole = pushSearch(level, WHOLE_STATE_CAP);
  if (whole !== null) return whole.moves;
  return '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
