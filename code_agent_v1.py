# code_agent.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Sequence
import random

Color = Literal["ROT", "BLAU", "GELB", "VIOLETT"]
CardKind = Literal["NUMBER", "ACTION", "CODE"]
ActionType = Literal["TAUSCH", "RICHTUNGSWECHSEL", "AUSSETZEN", "JOKER", "PLUS2", "GESCHENK", "RESET"]

@dataclass
class Card:
    """Represents a play card: NUMBER or ACTION. CODE handled via CodeCard."""
    kind: CardKind
    value: Optional[int] = None      # 0..9 for NUMBER
    color: Optional[Color] = None    # for NUMBER
    action: Optional[ActionType] = None  # for ACTION

@dataclass
class CodeCard:
    """Hidden code with four target digits."""
    target: Tuple[int, int, int, int]

@dataclass
class PlayerState:
    """Player hand and code."""
    hand: List[Card]
    code: CodeCard

@dataclass
class GameState:
    """All state required to play."""
    players: List[PlayerState]
    active_idx: int
    direction: int                # +1 clockwise, -1 counterclockwise
    draw_pile: List[Card]
    number_discard: List[Card]
    action_discard: List[Card]
    enable_reset: bool = True
    pending_plus2: int = 0        # total +2 to resolve at start of affected player's turn

# ---------- Level 1: Orchestration ----------

def play_human_vs_bot(num_players: int = 2) -> int:
    """Play a game with first player as human; returns winner index."""
    state = setup_game(num_players=num_players)
    winner = None
    while winner is None:
        state = take_turn(state, is_human=(state.active_idx == 0))
        state, winner = end_if_win(state)
    print(f"Winner: Player {winner}")
    return winner

def play_self_play(num_players: int = 2) -> int:
    """Bot vs bot; returns winner index."""
    state = setup_game(num_players=num_players)
    winner = None
    while winner is None:
        state = take_turn(state, is_human=False)
        state, winner = end_if_win(state)
    return winner

def take_turn(state: GameState, is_human: bool = False) -> GameState:
    """Execute a full turn for the active player, including PLUS2 resolution."""
    idx = state.active_idx
    player = state.players[idx]

    # Resolve pending PLUS2: allow counter with PLUS2 before drawing; otherwise forced draw and skip further play.
    if state.pending_plus2 > 0:
        plus2_in_hand = next((c for c in player.hand if c.kind == "ACTION" and c.action == "PLUS2"), None)
        if plus2_in_hand:
            move = ("PlayAction", (plus2_in_hand,))
            print(f"Player {idx} counters PLUS2.")
            state = apply_move(state, move[0], move[1])
        else:
            draws = state.pending_plus2
            print(f"Player {idx} draws {draws} due to PLUS2.")
            for _ in range(draws):
                state = apply_move(state, "Draw", ())
            state.pending_plus2 = 0
            # End turn after forced draws
            state.active_idx = next_player_index(state, steps=1)
            return state

    # Choose and apply move
    moves = legal_moves(state)
    if is_human:
        # Comprehensive display for human player
        print(f"\n{'='*60}")
        print(f"Your Code: {code_summary(player.code)}")
        print(f"Your Hand: {hand_summary(player)}")
        print(f"{'='*60}")
        
        # Show top discard cards
        top_num = state.number_discard[-1] if state.number_discard else None
        top_act = state.action_discard[-1] if state.action_discard else None
        print(f"Top Number Discard: {top_num.color} {top_num.value}" if top_num else "Top Number Discard: (empty)")
        print(f"Top Action Discard: {top_act.action}" if top_act else "Top Action Discard: (empty)")
        print(f"{'='*60}")
        
        # List all legal moves
        print(f"Legal moves for Player {idx}:")
        for mi, m in enumerate(moves):
            mt, payload = m
            label = describe_move(mt, payload)
            print(f"  {mi}: {label}")
        
        choice = 0
        try:
            raw = input("Choose move index (default 0): ").strip()
            if raw != "":
                choice = int(raw)
            if choice < 0 or choice >= len(moves):
                print(f"Invalid choice {choice}, using default 0.")
                choice = 0
        except (ValueError, EOFError, KeyboardInterrupt):
            print("Invalid input, using default 0.")
            choice = 0
        move = moves[choice]
    else:
        move = agent_decide_move(state)
        agent_move_log(state, move)

    if is_human:
        print(f"Player {idx} plays: {describe_move(move[0], move[1])}")
    state = apply_move(state, move[0], move[1])

    # Advance turn normally unless AUSSETZEN handled within apply_move
    if move[0] != "PlayAction" or (move[0] == "PlayAction" and move[1][0].action not in ("AUSSETZEN",)):
        state.active_idx = next_player_index(state, steps=1)
    return state

def end_if_win(state: GameState) -> Tuple[GameState, Optional[int]]:
    """Check if any player wins immediately after the last turn."""
    for i in range(len(state.players)):
        if is_win(state, i):
            return (state, i)
    return (state, None)

# ---------- Level 2: Helpers ----------

def setup_game(num_players: int, enable_reset: bool = True) -> GameState:
    """Create decks, deal codes and hands, and initialize piles."""
    # Build number cards: 0..9, two copies per value per color, four colors → 80
    colors: Sequence[Color] = ("ROT", "BLAU", "GELB", "VIOLETT")
    number_cards: List[Card] = []
    for color in colors:
        for v in range(10):
            number_cards.append(Card(kind="NUMBER", value=v, color=color))
            number_cards.append(Card(kind="NUMBER", value=v, color=color))

    # Build action cards (counts per prompt)
    actions_spec = {
        "TAUSCH": 6,
        "RICHTUNGSWECHSEL": 5,
        "AUSSETZEN": 6,
        "JOKER": 4,   # Note: Joker cannot be discarded; only for win in hand.
        "PLUS2": 4,
        "GESCHENK": 4,
        "RESET": 1,
    }
    action_cards: List[Card] = []
    for a, count in actions_spec.items():
        for _ in range(count):
            action_cards.append(Card(kind="ACTION", action=a))  # value/color None

    # Use random generator for all randomness
    rng = random.Random()  # Uses system time for true randomness
    
    # Generate unique code cards with 4 DIFFERENT digits each
    # Each code must have 4 distinct digits (0-9), and no two players get the same code
    code_cards: List[CodeCard] = []
    used_codes = set()
    
    while len(code_cards) < num_players:
        # Generate 4 different random digits
        digits = rng.sample(range(10), 4)  # sample ensures no duplicates
        code_tuple = tuple(digits)
        
        # Ensure this exact code hasn't been used
        if code_tuple not in used_codes:
            used_codes.add(code_tuple)
            code_cards.append(CodeCard(target=code_tuple))

    # Build draw pile: all play cards
    draw_pile = number_cards + action_cards
    rng.shuffle(draw_pile)

    # Assign codes to players (already unique)
    players: List[PlayerState] = []
    for i in range(num_players):
        players.append(PlayerState(hand=[], code=code_cards[i]))

    # Deal 7 play cards each
    for _ in range(7):
        for p in players:
            if not draw_pile:
                print("Warning: Not enough cards to deal initial hands.")
                break
            p.hand.append(draw_pile.pop())

    # Initialize number_discard: flip until a NUMBER is on top
    number_discard: List[Card] = []
    action_discard: List[Card] = []
    while draw_pile:
        top = draw_pile.pop()
        if top.kind == "NUMBER":
            number_discard.append(top)
            break
        else:
            # Action cards start their own discard when flipped
            action_discard.append(top)

    state = GameState(
        players=players,
        active_idx=0,
        direction=1,
        draw_pile=draw_pile,
        number_discard=number_discard,
        action_discard=action_discard,
        enable_reset=enable_reset,
        pending_plus2=0,
    )
    return state

def legal_moves(state: GameState) -> List[Tuple[str, Tuple]]:
    """Compute legal moves for active player. Returns list of (move_type, payload)."""
    moves: List[Tuple[str, Tuple]] = []
    player = state.players[state.active_idx]
    top_num = state.number_discard[-1] if state.number_discard else None

    # If pending PLUS2 on the active player, only counter with PLUS2 or forced draw handled in take_turn
    if state.pending_plus2 > 0:
        for c in player.hand:
            if c.kind == "ACTION" and c.action == "PLUS2":
                moves.append(("PlayAction", (c,)))
        if not moves:
            # No PLUS2 to counter; draw handled by take_turn, so offer Draw for consistency
            if state.draw_pile:
                moves.append(("Draw", ()))
            else:
                moves.append(("Pass", ()))
        return ordered_moves(moves)

    # PlayNumber: match color OR value with top number card
    if top_num:
        for c in player.hand:
            if c.kind == "NUMBER" and (c.color == top_num.color or c.value == top_num.value):
                moves.append(("PlayNumber", (c,)))

    # PlaySum: EXACTLY two number cards whose values sum to top.value; colors irrelevant; top of pair defines next
    if top_num:
        target_sum = top_num.value
        nums = [c for c in player.hand if c.kind == "NUMBER"]
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i], nums[j]
                if a.value is not None and b.value is not None and target_sum is not None:
                    if (a.value + b.value) == target_sum:
                        # Place in fixed order: (a, then b) for determinism
                        moves.append(("PlaySum", (a, b)))

    # PlayAction: any action can be played (JOKER excluded from discard)
    for c in player.hand:
        if c.kind == "ACTION" and c.action != "JOKER":
            # TAUSCH requires target stack selection; add both options if available
            if c.action == "TAUSCH":
                if state.number_discard:
                    moves.append(("PlayAction", (c, "number")))
                if state.action_discard:
                    moves.append(("PlayAction", (c, "action")))
            elif c.action == "GESCHENK":
                # Choose any opponent index
                for opp_idx in range(len(state.players)):
                    if opp_idx != state.active_idx:
                        moves.append(("PlayAction", (c, opp_idx)))
            elif c.action == "RESET" and state.enable_reset:
                for opp_idx in range(len(state.players)):
                    if opp_idx != state.active_idx:
                        moves.append(("PlayAction", (c, opp_idx)))
            else:
                moves.append(("PlayAction", (c,)))

    # HOUSE RULE: If no number/sum/action plays available, allow playing ANY number card
    # This prevents deadlock where players have no matching cards
    has_play_moves = len(moves) > 0
    if not has_play_moves:
        for c in player.hand:
            if c.kind == "NUMBER":
                moves.append(("PlayAny", (c,)))

    # Draw is always allowed if draw_pile not empty
    if state.draw_pile:
        moves.append(("Draw", ()))
    else:
        if not moves:
            moves.append(("Pass", ()))

    return ordered_moves(moves)

def apply_move(state: GameState, move_type: str, payload: Tuple) -> GameState:
    """Apply the selected move to the state and handle effects. Returns updated state."""
    idx = state.active_idx
    player = state.players[idx]

    def remove_card(card: Card):
        # Remove the specific card instance (by identity)
        for k, hcard in enumerate(player.hand):
            if hcard is card:
                player.hand.pop(k)
                return
        # If card not found, log error but continue
        print(f"Warning: Card {describe_move('PlayNumber' if card.kind == 'NUMBER' else 'PlayAction', (card,))} not found in hand.")

    if move_type == "PlayNumber" or move_type == "PlayAny":
        # PlayAny is like PlayNumber but allowed when no matches exist
        card: Card = payload[0]
        remove_card(card)
        state.number_discard.append(card)
        return reshuffle_if_needed(state)

    if move_type == "PlaySum":
        a: Card = payload[0]
        b: Card = payload[1]
        remove_card(a)
        remove_card(b)
        # Place a then b; b defines next color+value
        state.number_discard.append(a)
        state.number_discard.append(b)
        return reshuffle_if_needed(state)

    if move_type == "PlayAction":
        card: Card = payload[0]
        action = card.action
        # Special parameter for TAUSCH/GESCHENK/RESET
        remove_card(card)
        state.action_discard.append(card)

        if action == "TAUSCH":
            target_stack = payload[1] if len(payload) > 1 else "number"
            if target_stack == "number" and state.number_discard:
                player.hand.append(state.number_discard.pop())
            elif target_stack == "action" and state.action_discard[:-1]:
                # Take the previous top (avoid taking the TAUSCH just placed)
                taken = state.action_discard.pop(-2)
                player.hand.append(taken)

        elif action == "RICHTUNGSWECHSEL":
            state.direction *= -1

        elif action == "AUSSETZEN":
            # Skip next player by advancing +2 after this move; take_turn handles normal +1 increment,
            # so we set active_idx to skip right now and let caller refrain from normal increment.
            state.active_idx = next_player_index(state, steps=2)

        elif action == "PLUS2":
            state.pending_plus2 += 2

        elif action == "GESCHENK":
            target_idx = payload[1]
            # Reveal one card from each opponent (choose highest match potential)
            gift_card = choose_best_gift(state, target_idx)
            if gift_card:
                # Transfer chosen card; target draws a replacement
                opp = state.players[target_idx]
                # Remove from opponent hand by identity
                for k, hcard in enumerate(opp.hand):
                    if hcard is gift_card:
                        opp.hand.pop(k)
                        player.hand.append(gift_card)
                        break
                if state.draw_pile:
                    opp.hand.append(state.draw_pile.pop())

        elif action == "RESET":
            target_idx = payload[1]
            if state.enable_reset:
                # Replace target's code with a new one (simple deterministic next code)
                target = state.players[target_idx]
                new_target = CodeCard(target=tuple((x + 1) % 10 for x in target.code.target))
                target.code = new_target

        return reshuffle_if_needed(state)

    if move_type == "Draw":
        # Draw one card and add to hand. Turn ends.
        # NOTE: Auto-play disabled to prevent "moving target" deadlocks.
        # The agent will play the card on their next turn if it matches.
        if not state.draw_pile:
            print("Warning: Draw pile empty, cannot draw.")
            return state
        try:
            drawn = state.draw_pile.pop()
            player.hand.append(drawn)
        except IndexError:
            print("Error: Draw pile unexpectedly empty.")
            return state

        return reshuffle_if_needed(state)

    if move_type == "Pass":
        # Only valid if nothing else possible
        return state

    return state

def agent_decide_move(state: GameState) -> Tuple[str, Tuple]:
    """Deterministic, rule-based choice per priority hierarchy.
    KEY RULE: Keep at most ONE copy of each code-matching digit!
    """
    moves = legal_moves(state)
    idx = state.active_idx
    player = state.players[idx]
    hand_size = len(player.hand)
    current_matches = count_code_matches(player.hand, player.code)
    code_digits = set(player.code.target)
    
    # Count how many of each code digit we have in hand
    code_digit_counts = {d: 0 for d in code_digits}
    for c in player.hand:
        if c.kind == "NUMBER" and c.value in code_digits:
            code_digit_counts[c.value] += 1

    def should_protect(card: Card) -> bool:
        """Only protect a code-matching card if we have exactly 1 of that digit."""
        if card.kind != "NUMBER" or card.value not in code_digits:
            return False
        return code_digit_counts[card.value] <= 1
    
    def mark_as_played(card: Card):
        """Track that we're playing this card (for duplicate counting)."""
        if card.kind == "NUMBER" and card.value in code_digit_counts:
            code_digit_counts[card.value] -= 1

    # P0: Already winning (exactly 4 cards matching code)?
    if is_win(state, idx):
        return pick_by_priority(moves)

    # P1: Check for moves that lead to immediate win
    for mt, payload in moves:
        if mt in ("PlayNumber", "PlaySum", "PlayAny"):
            hyp_state = simulate_move(state, mt, payload)
            if is_win(hyp_state, idx):
                return (mt, payload)

    # P2: If we have 4+ matches and >4 cards, reduce hand but protect ONE copy of each code digit!
    if current_matches >= 4 and hand_size > 4:
        best_move = None
        best_score = -9999
        for mt, payload in moves:
            if mt in ("PlayNumber", "PlaySum", "PlayAny", "PlayAction"):
                # Skip moves that would play a protected code card (only ONE copy protected)
                if mt in ("PlayNumber", "PlayAny") and should_protect(payload[0]):
                    continue
                if mt == "PlaySum":
                    if should_protect(payload[0]) or should_protect(payload[1]):
                        continue
                hyp_state = simulate_move(state, mt, payload)
                new_hand = len(hyp_state.players[idx].hand)
                new_matches = count_code_matches(hyp_state.players[idx].hand, player.code)
                hand_reduction = hand_size - new_hand
                progress_loss = current_matches - new_matches
                score = hand_reduction * 10 - progress_loss * 5
                if new_hand == 4 and new_matches >= 4:
                    score += 100
                if score > best_score:
                    best_score = score
                    best_move = (mt, payload)
        if best_move:
            return best_move

    # P3: Play number/sum that DOESN'T lose code matches
    for mt, payload in moves:
        if mt in ("PlayNumber", "PlaySum", "PlayAny"):
            # Skip if we'd play a protected code card (keep ONE copy of each digit)
            if mt in ("PlayNumber", "PlayAny") and should_protect(payload[0]):
                continue
            if mt == "PlaySum":
                if should_protect(payload[0]) or should_protect(payload[1]):
                    continue
            hyp_state = simulate_move(state, mt, payload)
            new_matches = count_code_matches(hyp_state.players[idx].hand, player.code)
            if new_matches >= current_matches:
                return (mt, payload)

    # P4: Play any NUMBER/SUM that doesn't require playing protected code cards
    # Prioritize SUM (reduces by 2) over NUMBER (reduces by 1)
    for mt, payload in moves:
        if mt == "PlaySum":
            if not should_protect(payload[0]) and not should_protect(payload[1]):
                return (mt, payload)
    for mt, payload in moves:
        if mt == "PlayNumber":
            if not should_protect(payload[0]):
                return (mt, payload)

    # P5: Action card preferences: PLUS2 > AUSSETZEN > RICHTUNGSWECHSEL > others
    for pref in ("PLUS2", "AUSSETZEN", "RICHTUNGSWECHSEL", "GESCHENK", "TAUSCH"):
        for mt, payload in moves:
            if mt == "PlayAction" and payload[0].action == pref:
                return (mt, payload)

    # P6: Any remaining action card
    for mt, payload in moves:
        if mt == "PlayAction":
            return (mt, payload)

    # P7: PlayAny - but try to avoid protected code cards first
    for mt, payload in moves:
        if mt == "PlayAny" and not should_protect(payload[0]):
            return (mt, payload)
    
    # P8: If ALL playable cards are protected, we must play one anyway to avoid infinite draw
    for mt, payload in moves:
        if mt in ("PlayNumber", "PlaySum", "PlayAny"):
            return (mt, payload)

    # P9: Draw as last resort
    return pick_by_priority(moves)

def is_win(state: GameState, player_idx: int) -> bool:
    """Win if EXACTLY 4 hand cards match code values; Joker allowed to fill only for matching."""
    hand = state.players[player_idx].hand
    code = state.players[player_idx].code
    # Only NUMBER and JOKER in hand count toward match. Must have exactly 4 cards in hand.
    if len(hand) != 4:
        return False
    # Count how many of the four positions can be satisfied by NUMBER with same value; Joker satisfies any single missing value.
    values = [c.value for c in hand if c.kind == "NUMBER"]
    jokers = sum(1 for c in hand if c.kind == "ACTION" and c.action == "JOKER")
    target = list(code.target)

    # Greedy match: remove matched values from target using number cards
    for v in list(values):
        if v in target:
            target.remove(v)
            values.remove(v)

    # Use jokers to fill remaining positions
    if jokers >= len(target):
        return True
    return False

def count_code_matches(hand: List[Card], code: CodeCard) -> int:
    """Count how many target digits can be covered by current hand (NUMBER matches + JOKER fillers)."""
    target = list(code.target)
    values = [c.value for c in hand if c.kind == "NUMBER"]
    jokers = sum(1 for c in hand if c.kind == "ACTION" and c.action == "JOKER")
    matches = 0
    # Match with numbers
    for v in list(values):
        if v in target:
            target.remove(v)
            matches += 1
    # Fill with jokers
    fill = min(jokers, len(target))
    matches += fill
    return matches

def reshuffle_if_needed(state: GameState) -> GameState:
    """If draw pile empty, reshuffle from discards keeping top of each pile."""
    if state.draw_pile:
        return state
    
    # Keep top of number and action discards
    keep_nums = state.number_discard[-1:] if state.number_discard else []
    keep_actions = state.action_discard[-1:] if state.action_discard else []
    new_draw: List[Card] = []
    # Collect all but top from number discard
    new_draw.extend(state.number_discard[:-1])
    # Collect all but top from action discard
    new_draw.extend(state.action_discard[:-1])
    
    if not new_draw:
        print("Warning: No cards available to reshuffle.")
        return state
    
    # Shuffle with evolving seed
    seed = 123 + len(new_draw) + len(state.players) + state.active_idx
    rng = random.Random(seed)
    rng.shuffle(new_draw)
    state.draw_pile = new_draw
    # Reset discards to kept tops
    state.number_discard = keep_nums[:]
    state.action_discard = keep_actions[:]
    return state

def next_player_index(state: GameState, steps: int = 1) -> int:
    """Compute next player index given direction and steps."""
    n = len(state.players)
    return (state.active_idx + steps * state.direction) % n

# ---------- Inline utilities (remain Level-2 usage) ----------

def describe_move(move_type: str, payload: Tuple) -> str:
    """Human-readable move label."""
    if move_type == "PlayNumber":
        c = payload[0]
        return f"PlayNumber {c.color} {c.value}"
    if move_type == "PlaySum":
        a, b = payload
        return f"PlaySum ({a.color} {a.value}) + ({b.color} {b.value})"
    if move_type == "PlayAction":
        c = payload[0]
        extra = ""
        if len(payload) > 1:
            extra = f" {payload[1]}"
        return f"PlayAction {c.action}{extra}"
    if move_type == "Draw":
        return "Draw"
    return "Pass"

def ordered_moves(moves: List[Tuple[str, Tuple]]) -> List[Tuple[str, Tuple]]:
    """Deterministic order: Action > Sum > Number > Draw > Pass; secondary lexicographic."""
    def key(m):
        mt, payload = m
        pri = {"PlayAction": 0, "PlaySum": 1, "PlayNumber": 2, "Draw": 3, "Pass": 4}.get(mt, 9)
        # Secondary: color, value, action string stable sort
        col = ""
        val = -1
        act = ""
        if mt == "PlayNumber":
            c = payload[0]
            col = c.color or ""
            val = c.value or -1
        elif mt == "PlaySum":
            a, b = payload
            col = (b.color or "")  # top is second
            val = b.value or -1
        elif mt == "PlayAction":
            c = payload[0]
            act = c.action or ""
        return (pri, act, col, val)
    return sorted(moves, key=key)

def pick_by_priority(moves: List[Tuple[str, Tuple]]) -> Tuple[str, Tuple]:
    """Pick the first move in deterministic priority order."""
    ordered = ordered_moves(moves)
    return ordered[0] if ordered else ("Pass", ())

def better_lex(mt: str, payload: Tuple, current_best: Optional[Tuple[str, Tuple]]) -> bool:
    """Tie-breaker using ordered_moves key comparison."""
    if current_best is None:
        return True
    return ordered_moves([(mt, payload), current_best])[0] == (mt, payload)

def simulate_move(state: GameState, mt: str, payload: Tuple) -> GameState:
    """Shallow simulation: copy minimal structures to evaluate heuristics."""
    # Copy shallow structures to avoid identity issues
    players = []
    for i, p in enumerate(state.players):
        players.append(PlayerState(hand=list(p.hand), code=CodeCard(target=tuple(p.code.target))))
    sim = GameState(
        players=players,
        active_idx=state.active_idx,
        direction=state.direction,
        draw_pile=list(state.draw_pile),
        number_discard=list(state.number_discard),
        action_discard=list(state.action_discard),
        enable_reset=state.enable_reset,
        pending_plus2=state.pending_plus2,
    )
    return apply_move(sim, mt, payload)

def choose_best_gift(state: GameState, target_idx: int) -> Optional[Card]:
    """Pick one card from target opponent to transfer; preference for NUMBER that helps code."""
    opp = state.players[target_idx]
    # Prefer NUMBER that contributes; else any NUMBER; else first card
    for c in opp.hand:
        if c.kind == "NUMBER" and contributes_to_code(c, state.players[state.active_idx].code):
            return c
    for c in opp.hand:
        if c.kind == "NUMBER":
            return c
    return opp.hand[0] if opp.hand else None

def contributes_to_code(card: Card, code: CodeCard) -> bool:
    """True if card's value can match one of the remaining target digits."""
    if card.kind != "NUMBER" or card.value is None:
        return False
    return card.value in code.target

def hand_summary(p: PlayerState) -> str:
    """Readable summary of a player's hand."""
    parts = []
    for c in p.hand:
        if c.kind == "NUMBER":
            parts.append(f"{c.color} {c.value}")
        elif c.kind == "ACTION":
            parts.append(f"ACTION {c.action}")
        else:
            parts.append("UNKNOWN")
    return ", ".join(parts)

def code_summary(code: CodeCard) -> str:
    """Readable summary of a player's hidden code."""
    a, b, c, d = code.target
    return f"{a}-{b}-{c}-{d}"

def agent_move_log(state: GameState, move: Tuple[str, Tuple]) -> None:
    """Log the agent's move decision."""
    idx = state.active_idx
    label = describe_move(move[0], move[1])
    print(f"[Bot Player {idx}] Move: {label}")

if __name__ == "__main__":
    import sys

    mode = "human"
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("human", "bot"):
        mode = sys.argv[1].lower()

    num_players = 2
    
    try:
        state = setup_game(num_players=num_players)

        print("=== CODE Game ===")
        print(f"Mode: {'Human vs Bot' if mode == 'human' else 'Bot vs Bot'}")
        print(f"Players: {num_players}")
        print(f"Starting top number: {state.number_discard[-1].color} {state.number_discard[-1].value}")

        winner = None
        turn_count = 0
        max_turns = 500  # Prevent infinite loops
        
        while winner is None and turn_count < max_turns:
            idx = state.active_idx
            top = state.number_discard[-1] if state.number_discard else None
            print(f"\n--- Player {idx}'s turn ---")
            if top:
                print(f"Top number: {top.color} {top.value}")
            print(f"Hand size: {len(state.players[idx].hand)} | Pending +2: {state.pending_plus2}")

            is_human = (mode == "human" and idx == 0)
            state = take_turn(state, is_human=is_human)
            state, winner = end_if_win(state)
            turn_count += 1

        if winner is not None:
            print(f"\n=== Winner: Player {winner} ===")
        else:
            print(f"\n=== Game ended after {max_turns} turns (limit reached) ===")
            
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()