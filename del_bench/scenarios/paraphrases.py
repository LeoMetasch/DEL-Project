"""Hand-authored narrative paraphrases for registered scenarios.

Each entry supplies variants 1 and 2. The canonical scenario narrative remains
variant 0 in the scenario module itself.
"""

from __future__ import annotations

from typing import Dict, Tuple

from del_bench.scenarios.schema import clean_text

PARAPHRASES: Dict[str, Tuple[str, str]] = {
    "coin_public_peek_v1": (
        clean_text("""
            Alice and Bob are in a referee-run coin game.
            They both know the rules well.
            A face-down coin is hidden under an opaque cup, and neither player has seen the face yet.
            From the referee's usual habits, both currently believe the coin is heads-up.
            The coin is actually heads-up.

            The referee briefly raises the cup and gives Bob visual access to the coin.
            Alice can see that Bob is looking, but she cannot see the coin's face herself.
            The cup is then put back.
            Alice and Bob both understand that Bob has now seen the coin while Alice has not seen the result.
            """),
        clean_text("""
            Alice and Bob are playing a coin game whose rules are common ground between them.
            The coin is covered by an opaque cup.
            Neither has observed the coin's face, and both start with the settled belief that it is heads-up because of the referee's known habits.
            In reality, the coin is heads-up.

            The referee lifts the cup for Bob.
            Bob sees the face of the coin.
            Alice observes that Bob is being shown the coin, but she does not see which side is up.
            Afterward, it is common ground between Alice and Bob that Bob saw the coin and Alice did not see the face.
            """),
    ),
    "coin_private_peek_v1": (
        clean_text("""
            Alice and Bob are taking part in a coin game with a referee.
            The rules are familiar to both of them and forbid peeking without permission.
            A face-down coin is covered by an opaque cup.
            Neither player has seen the face yet.
            Because of the referee's usual habits, both believe the coin is heads-up, and it is in fact heads-up.

            Alice leaves the room for a short time.
            During that interval, Bob lifts the cup, sees that the coin is heads-up, and puts the cup back.
            Alice is absent for the whole episode and receives no evidence that Bob looked.
            Since Bob is known to follow the rules, Alice has no reason to suspect he peeked, and it is mutually understood that she trusts him to play by the rules.
            Bob says nothing about the peek.
            Alice then returns.
            """),
        clean_text("""
            Background: Alice and Bob both know the coin-game rules, including the rule against unauthorized peeking.
            The coin is under an opaque cup.
            Neither has yet seen whether it is heads or tails.
            Both are inclined to believe heads because of the referee's habits, and heads is the actual state.

            While Alice is out of the room, Bob checks the coin under the cup and sees heads.
            He replaces the cup before Alice comes back.
            Alice does not witness the action, is given no sign that it occurred, and has no reason to think Bob would break the rules.
            Bob remains silent about what he did.
            Alice returns to the room.
            """),
    ),
    "coin_successful_lie_v1": (
        clean_text("""
            Alice and Bob are in a referee-run coin game.
            Both know the rules, including the rule against peeking without permission.
            A covered coin lies on the table.
            Neither player has initially seen the face.
            Based on the referee's habits, both believe the coin is heads-up, and the coin really is heads-up.

            While Alice is briefly outside the room, Bob lifts the cup and sees that the coin is heads-up.
            He replaces the cup before Alice returns.
            Alice does not see this, receives no evidence of it, and has no reason to suspect Bob, who is understood to be rule-abiding.

            After Alice returns, Bob says aloud to her, "I am certain the coin is tails-up." Alice has not seen the coin and has no other information.
            She fully trusts Bob's statement, and both players understand that Alice accepts Bob's word.
            """),
        clean_text("""
            Alice and Bob share the rules of the coin game, and those rules prohibit unauthorized peeking.
            The coin is hidden under an opaque cup.
            Before any action, neither player has seen it.
            Both believe heads because of the referee's known pattern; heads is also the actual face.

            Bob then obtains private information.
            Alice is away from the room, and Bob looks under the cup.
            He sees heads and covers the coin again.
            Alice neither observes this nor has any reason to think it happened.

            Later, Bob speaks in Alice's presence.
            He confidently tells her that he is certain the coin is tails-up.
            Alice lacks independent access to the coin and treats Bob as fully reliable, so the exchange is mutually understood as one in which Alice trusts Bob's claim.
            """),
    ),
    "coin_moore_sentence_v1": (
        clean_text("""
            Alice and Bob are playing the coin game under a referee's rules.
            A coin lies under an opaque cup on the table.

            Before the round, the referee privately showed the coin to Alice alone.
            Alice therefore knows which side is showing.
            The coin is actually heads-up.
            Bob has not seen the coin.
            From the way the referee set up this round and from earlier habits, Bob's current settled belief is that the coin is tails-up.

            Alice then speaks to Bob where he can clearly hear her: "The coin is heads-up, and you do not currently believe the coin is heads-up." Bob completely trusts Alice's truthfulness about the coin and accepts what she has just said.
            Both understand that Alice made this statement and that Bob accepted it.
            """),
        clean_text("""
            The rules of the referee's coin game are common ground for Alice and Bob.
            The coin is covered on the table.

            Only Alice has been shown the coin by the referee before play begins, so Alice knows the face.
            The actual face is heads.
            Bob has not had visual access to the coin; given the setup and the referee's earlier habits, Bob presently believes tails.

            Alice publicly addresses Bob and says that the coin is heads and that Bob does not currently believe it is heads.
            Bob regards Alice as completely truthful about the coin and accepts her utterance.
            It is shared between them that she spoke and that he accepted the statement.
            """),
    ),
    "coin_truthful_announcement_v1": (
        clean_text("""
            Alice and Bob are playing a coin game whose rules both know well; unauthorized peeking is not allowed.
            The coin is face-down beneath an opaque cup.
            Initially neither player has seen it.
            Both believe heads because of the referee's habits, and heads is the actual state.

            Alice briefly leaves the room.
            While she is away, Bob lifts the cup, sees that the coin is heads-up, and covers it again.
            Alice does not witness this and has no indication that Bob broke the rules; Bob is understood to be someone she trusts to follow them.

            When Alice returns, Bob says aloud, "I am certain the coin is heads-up." Alice has not seen the coin and has no other evidence.
            She fully trusts Bob's word, and both understand that she accepts his statement.
            """),
        clean_text("""
            At the start, Alice and Bob share the same background: the coin is hidden under a cup, neither has seen it, and both believe it is heads because of the referee's known habits.
            The rules also make clear that peeking without permission is forbidden.
            The coin is indeed heads.

            Bob then gets private visual evidence while Alice is out of the room.
            He looks under the cup, sees heads, and replaces the cup.
            Alice is unaware of this event and has no reason to suspect it.

            Afterward, in Alice's presence, Bob confidently claims that the coin is heads-up.
            Alice has no independent source about the coin, trusts Bob completely, and the two mutually understand that this trusted announcement has occurred.
            """),
    ),
    "coin_peek_then_announce_v1": (
        clean_text("""
            Alice and Bob are in a referee-run coin game.
            They both know the rules.
            The coin is covered by an opaque cup, and neither player has seen its face.
            Both believe heads because of the referee's usual habits.
            The coin is actually heads-up.

            The referee lifts the cup and shows the coin to Bob.
            Alice can see that Bob is looking at the coin, but she cannot see the face herself.
            The cup is replaced.
            Alice and Bob both understand that Bob has seen the coin while Alice has not.

            Alice then says aloud to Bob, "I do not know which side the coin is showing." Both Alice and Bob understand that Alice has made this true statement.
            """),
        clean_text("""
            Initial conditions: the rules are common ground for Alice and Bob; the coin is under an opaque cup; neither has observed the face; both initially believe heads; heads is the actual face.

            The referee gives Bob a look at the coin.
            Alice observes Bob receiving that look but not the result.
            Once the cup is replaced, it is common ground that Bob saw the coin and Alice did not see which side was up.

            Alice publicly tells Bob that she does not know which side is showing.
            The statement is true, and both players mutually understand that she has said it.
            """),
    ),
    "coin_soft_announcement_v1": (
        clean_text("""
            Alice and Bob are playing the coin game with a referee.
            They both know the rules.
            A face-down coin is hidden under an opaque cup.
            Neither player has seen the face.
            Because of the referee's known habits, both believe the coin is heads-up, and it is actually heads-up.

            A bystander who sometimes watches these games passes by and says out loud, "I think the coin is tails-up this time." Alice and Bob both hear the remark.
            They know this bystander is neither fully reliable nor useless: the bystander is sometimes right and sometimes wrong.
            They do not treat the comment as authoritative, but they do give it some weight, and this is mutually understood.

            The remark does not rule out heads.
            The bystander may be wrong.
            Still, the comment shifts Alice and Bob's inclination toward tails.
            """),
        clean_text("""
            The covered coin has not been seen by either Alice or Bob.
            Both begin with a heads belief based on the referee's habits, and heads is the actual state.

            A familiar onlooker says that the coin might be tails-up.
            Alice and Bob hear the same statement.
            They regard the onlooker as an imperfect source: not authoritative, but not irrelevant either.
            It is common ground that both heard the suggestion and assign it some evidential force.

            The suggestion is defeasible.
            Alice and Bob still consider heads possible, but after hearing the onlooker they lean more toward tails than before.
            """),
    ),
    "coin_unreliable_announcement_v1": (
        clean_text("""
            Alice and Bob are in the referee's coin game.
            The rules are known to both.
            The coin is under an opaque cup, and neither player has seen which side is up.
            From the referee's habits, both believe heads; the coin is actually heads.

            A stranger whom neither player knows walks past and says aloud, "The coin is tails-up." Alice and Bob both hear the statement.
            They also both know that the stranger has no information about the coin and is completely unreliable, making random guesses with no reason to be trusted.
            It is mutually understood that neither Alice nor Bob gives the stranger's claim any trust.

            The stranger then leaves.
            """),
        clean_text("""
            The coin remains covered, and neither Alice nor Bob has seen its face.
            Both initially believe heads because of the referee's usual habits.
            Heads is in fact the true state.

            An unknown passerby announces that the coin is tails.
            The announcement is heard clearly by both Alice and Bob.
            However, the passerby is known to have no access to the coin and to make unguarded random guesses.
            Alice and Bob therefore treat the announcement as uninformative, and this lack of trust is common ground between them.

            After making the claim, the passerby departs.
            """),
    ),
    "coin_trusted_lie_v1": (
        clean_text("""
            Alice and Bob are playing the referee's coin game.
            They know the rules, and the coin is hidden under an opaque cup.
            Neither Alice nor Bob has initially seen the face.
            Because of the referee's habits, both believe the coin is heads-up.
            The coin is actually heads-up.

            The referee briefly lifts the cup for Bob.
            Alice sees that Bob is looking at the coin but cannot see the face herself.
            Bob sees heads, the cup is replaced, and both players understand that Bob has observed the coin while Alice has not seen the result.

            Bob then says aloud to Alice, "I am certain the coin is tails-up." Alice has not seen the coin and has no other evidence.
            She fully trusts Bob's word and is satisfied by his statement.
            Both understand that Alice accepts Bob's claim.
            """),
        clean_text("""
            Alice and Bob initially lack visual access to the coin under the cup.
            The referee's usual habits lead both to believe heads, and heads is the real face.

            The referee shows Bob the covered coin's face while Alice watches Bob receive the observation but cannot see the result herself.
            Bob sees heads.
            It is common ground that Bob has now seen the coin and Alice has not.

            Bob then makes a confident statement to Alice that he is certain the coin is tails-up.
            Alice, however, has no independent information about the face and treats Bob as fully trustworthy.
            It is mutually understood that the statement occurred and that Alice trusts it.
            """),
    ),
    "coin_public_peek_tails_v1": (
        clean_text("""
            Alice and Bob are in a referee-run coin game.
            Both know the rules.
            The coin is covered by an opaque cup.
            Neither player has seen the face yet.
            Because of the referee's usual habits, both believe the coin is heads-up, but the coin is actually tails-up.

            The referee briefly lifts the cup for Bob.
            Bob sees the coin's face.
            Alice sees that Bob is looking at the coin, but she cannot see which side is up.
            The referee covers the coin again.
            Alice and Bob both understand that Bob saw the coin and Alice did not see the face.
            """),
        clean_text("""
            Initial conditions: the covered coin has not been observed by Alice or Bob.
            The shared prior expectation, based on the referee's habits, is heads.
            In reality, the coin is tails.

            The referee raises the cup and shows the coin to Bob.
            Alice observes that Bob receives this view but does not herself see the result.
            Once the cup is replaced, it is common ground between Alice and Bob that Bob has seen the coin while Alice has not seen its face.
            """),
    ),
    "coin_peek_then_soft_v1": (
        clean_text("""
            Alice and Bob are playing a coin game whose rules forbid unauthorized peeking.
            The coin is under an opaque cup.
            Neither player has seen the face at the start.
            Both believe heads because of the referee's habits, and heads is the actual face.

            Alice leaves briefly.
            While she is away, Bob lifts the cup, sees that the coin is heads-up, and covers it again.
            Alice does not see this and receives no evidence that it happened.
            Because Bob is trusted to follow the rules, Alice has no reason to suspect him.
            Bob says nothing.

            After Alice returns, a bystander says aloud that they think the coin is tails-up this time.
            Alice and Bob both hear this.
            The bystander is only partly reliable: not authoritative, but worth some consideration.
            Both players understand that Alice gives the statement some weight but still cannot rule out heads.
            The comment shifts Alice's inclination toward tails. Because she remains unaware that Bob peeked, she expects him to shift too, and Bob understands that. In fact, his direct observation keeps his knowledge and belief fixed on heads.
            """),
        clean_text("""
            At first, Alice and Bob both lack direct evidence about the coin under the cup.
            They both believe heads from the referee's habits, and heads is true.
            The rules make unauthorized peeking forbidden.

            Bob then privately learns the face while Alice is out of the room: he looks under the cup, sees heads, and restores the cup.
            Alice neither witnesses this nor has any reason to think it occurred; Bob gives no report of the peek.

            Later, with Alice back in the room, an occasional bystander says that the coin may be tails-up.
            Both Alice and Bob hear it.
            Alice treats the bystander as a fallible source whose suggestion matters somewhat, so tails becomes more plausible for her while heads remains possible.
            Because Alice remains unaware of Bob's peek, she expects him to respond as she does, and Bob understands that. In fact, his direct observation keeps his knowledge and belief fixed on heads.
            """),
    ),
    "coin_private_peek_3agent_v1": (
        clean_text("""
            Alice, Bob, and Charles are in a referee-run coin game.
            All three know the rules, including the rule against peeking without permission.
            A face-down coin is hidden under an opaque cup.
            None of the three has seen the face yet.
            The referee's habits lead all three to believe heads, and heads is the actual state.

            Alice and Charles briefly leave the room.
            During their absence, Bob lifts the cup, sees that the coin is heads-up, and replaces the cup.
            Neither Alice nor Charles witnesses this or receives any evidence of it.
            Since Bob is understood to follow the rules, Alice and Charles have no reason to suspect him, and it is common ground among the three that they trust Bob to play by the rules.
            Bob says nothing about the peek.

            Alice and Charles then return.
            """),
        clean_text("""
            The coin is covered, and Alice, Bob, and Charles have not seen whether it is heads or tails.
            All three share the belief that it is heads because of the referee's habits.
            The coin really is heads.
            The rules prohibit unauthorized peeking.

            Bob checks the coin while Alice and Charles are away.
            He sees heads and restores the cup.
            The other two are absent for the entire event and have no indication that Bob looked.
            Because Bob is trusted as rule-abiding, neither Alice nor Charles suspects the peek, and Bob does not disclose it.

            Alice and Charles come back into the room.
            """),
    ),
    "coin_successful_lie_3agent_v1": (
        clean_text("""
            Alice, Bob, and Charles are playing the referee's coin game.
            The rules are common ground and forbid unauthorized peeking.
            A covered coin lies on the table.
            None of the players has initially seen the face.
            All three believe heads from the referee's habits, and the coin is actually heads.

            Alice and Charles leave briefly.
            While they are gone, Bob lifts the cup, sees heads, and covers the coin again.
            Alice and Charles do not witness this and receive no sign that it happened.
            They trust Bob to follow the rules.

            When Alice and Charles return, Bob says aloud to both of them, "I am certain the coin is tails-up." Neither Alice nor Charles has seen the coin or has any other source about it.
            Both fully trust Bob's word.
            It is mutually understood among all three that this exchange has occurred and that Alice and Charles accept Bob's claim.
            """),
        clean_text("""
            Alice, Bob, and Charles all know the rules of the game, and those rules disallow peeking without permission.
            The coin is under a cup.
            None of the three has seen the face.
            All begin with a heads belief based on the referee's habits; heads is true.

            Bob privately observes the coin.
            Alice and Charles are out of the room, Bob looks under the cup, sees heads, and puts the cup back.
            The others have no evidence of this and no reason to suspect Bob.

            Bob later makes a public claim to Alice and Charles that the coin is tails-up.
            They have no independent access to the coin and treat Bob as fully reliable.
            All three understand that Alice and Charles trust Bob's statement.
            """),
    ),
    "coin_double_peek_v1": (
        clean_text("""
            Alice and Bob are playing a coin game with a referee.
            Both know the rules, and the rules forbid unauthorized peeking.
            The coin is hidden under an opaque cup.
            Neither player has seen the face at the start.
            Both believe heads from the referee's habits, and the coin is actually heads.

            First Alice leaves the room.
            While she is away, Bob lifts the cup, sees heads, and covers the coin again.
            Alice is unaware of this, has no reason to suspect it, and Bob says nothing.
            Alice returns.

            Later Bob leaves the room.
            During Bob's absence, Alice lifts the cup, sees heads, and replaces the cup.
            Bob does not witness this and has no reason to suspect Alice, who is also trusted to follow the rules.
            Alice says nothing.
            Bob then returns.
            """),
        clean_text("""
            Initial conditions: Alice and Bob both know the game rules, the coin is covered, neither has seen the face, and both initially believe heads.
            Heads is the actual face.

            Bob's private peek comes first.
            Alice is outside the room, so Bob checks the coin, sees heads, and restores the cup without Alice learning that he did so.
            Alice returns with no evidence of Bob's peek.

            Alice's private peek comes second.
            Bob steps out, and Alice checks the same covered coin.
            She sees heads and puts the cup back.
            Bob does not see this and receives no sign of it.
            Alice remains silent about her peek, and Bob returns.
            """),
    ),
    "coin_lie_then_peek_v1": (
        clean_text("""
            Alice and Bob are in a referee-run coin game.
            Both know the rules, including the rule against unauthorized peeking.
            The coin is under an opaque cup.
            Neither player has initially seen the face.
            Both believe heads because of the referee's habits, and heads is true.

            While Alice is out of the room, Bob lifts the cup and sees that the coin is heads-up.
            He replaces the cup before Alice returns.
            Alice does not witness this and has no reason to suspect it.

            After Alice returns, Bob says aloud that he is certain the coin is tails-up.
            Alice has no independent information about the coin and fully trusts Bob's word, so both understand that she accepts the claim.

            Then the referee announces a coin check.
            In view of both players, the referee lifts the cup and shows the coin to Alice.
            Alice sees heads.
            Bob sees that Alice has looked, but he cannot see the face.
            The cup is replaced, and both understand that Alice has observed the result while Bob has not seen it.
            """),
        clean_text("""
            At the start, Alice and Bob share the same background: the covered coin has not been seen, both believe heads from the referee's habits, and the rules prohibit peeking.
            The actual face is heads.

            Bob's private observation comes first.
            Alice is away, Bob looks under the cup, sees heads, and hides the evidence again by replacing the cup.
            Alice is unaware of this.

            Bob's trusted false claim comes next.
            In Alice's presence, he confidently says the coin is tails.
            Alice has not seen the coin and trusts Bob completely, and the exchange is mutually understood.

            Alice's later observation comes last.
            The referee publicly allows Alice to check the coin.
            Alice sees heads; Bob sees that Alice checked, but not which side she saw.
            Both understand that Alice now has direct evidence and Bob does not see her result.
            """),
    ),
    "coin_conflicting_claims_v1": (
        clean_text("""
            Alice, Bob, and Charles are playing the referee's coin game.
            All three know the rules.
            The coin is hidden under an opaque cup, and none of them has seen its face.
            Because of the referee's habits, all three believe heads.
            The coin is actually heads.

            The referee briefly shows the coin to Bob while Alice and Charles watch Bob receive the observation without seeing the face themselves.
            Bob sees heads.
            All three understand that Bob observed the coin while Alice and Charles did not see his result.

            Bob speaks first, saying aloud to Alice and Charles, "I am certain the coin is tails-up." Alice and Charles both hear him.
            Bob is treated as knowledgeable and trustworthy about the game, so both of them fully trust his word.
            All three understand that this first exchange has occurred.

            The referee then briefly shows the coin to Charles while Alice and Bob watch Charles receive the observation without seeing the face themselves.
            Charles sees heads.
            All three understand that Charles observed the coin while Alice and Bob did not see his result.

            Charles then says aloud to Alice and Bob, "I am certain the coin is heads-up." Alice and Bob both hear Charles.
            Charles is treated as equally knowledgeable and trustworthy, and both Alice and Bob fully trust his word.
            It is common ground among the three that both claims have been made and heard.
            """),
        clean_text("""
            Alice, Bob, and Charles have not observed the coin under the cup.
            All three begin with a heads belief based on the referee's habits, and heads is the actual state.

            Bob receives a referee-authorized look at the coin while Alice and Charles watch him receive it without seeing the result.
            Bob sees heads, and this distribution of information is common ground.

            Bob publicly claims that the coin is tails.
            Alice and Charles hear the claim and regard Bob as fully reliable and knowledgeable in this game.
            It is mutually understood that they trust Bob's statement.

            Charles then receives his own referee-authorized look while Alice and Bob watch without seeing the result.
            Charles sees heads, and this distribution of information is also common ground.

            Charles later publicly claims that the coin is heads.
            Alice and Bob hear this later claim and regard Charles as just as reliable and knowledgeable as Bob.
            Everyone understands that both incompatible claims have now been heard by the group.
            """),
    ),
    "coin_witness_deception_v1": (
        clean_text("""
            Alice, Bob, and Charles are playing a referee-run coin game.
            All three know the rules.
            A face-down coin is covered by an opaque cup.
            None of the players has seen the face at the start.
            All three believe heads because of the referee's habits, and heads is the actual face.

            The referee first allows Bob to check the coin.
            In view of the group, the cup is lifted for Bob.
            Bob sees heads.
            Alice and Charles see that Bob looked, but neither sees the face.
            Everyone understands that Bob observed the coin and the others did not see his result.

            The referee then allows Charles to check.
            Again in view of the group, the cup is lifted for Charles.
            Charles sees heads.
            Alice and Bob see that Charles looked, but neither sees which side Charles saw.
            Everyone understands that Charles observed the coin and the others did not see his result.

            Bob then says aloud to Alice and Charles that he is certain the coin is tails-up.
            Alice has not seen the coin and has no other source, so she fully trusts Bob's word.
            Charles has seen the coin himself.
            All three understand that Bob's statement has just been made.
            """),
        clean_text("""
            The coin is under a cup, Alice, Bob, and Charles all begin without seeing the face, and all believe heads from the referee's habits.
            Heads is true.

            Bob receives the first authorized look.
            The referee shows the coin to Bob while Alice and Charles watch Bob being shown it.
            Bob sees heads; Alice and Charles do not see the face.
            This distribution of information is common ground.

            Charles receives the second authorized look.
            The referee shows the coin to Charles while Alice and Bob watch Charles being shown it.
            Charles sees heads; Alice and Bob do not see his result.
            This is also common ground.

            Finally, Bob publicly claims that the coin is tails.
            Alice lacks direct evidence and accepts Bob's word.
            Charles, unlike Alice, has directly seen heads.
            The group mutually understands that Bob made the claim in Alice's and Charles's presence.
            """),
    ),
    "coin_conditional_belief_v1": (
        clean_text("""
            Alice and Bob are in a referee-run coin game.
            The referee hides a coin under an opaque cup and seals a card in an envelope.
            The concealed card can be plain or marked.
            Neither player has observed the coin or the card.

            Both players know the referee's usual pattern.
            Overall, the most likely arrangement is a heads-up coin paired with a plain card.
            If the card is marked, tails is more likely than heads.
            If the card is plain, heads is more likely than tails.
            Alice and Bob reason from the same pattern.

            The actual arrangement is a heads-up coin and a plain card.
            """),
        clean_text("""
            A referee places a coin beneath an opaque cup and puts a card into a sealed envelope.
            Alice and Bob can see neither the coin's face nor whether the card is plain or marked.

            The referee's habits are familiar to both players.
            The single most likely setup is heads with a plain card.
            Within the marked-card setups, tails is the more likely coin face.
            Within the plain-card setups, heads is the more likely coin face.
            Alice and Bob both use this information when deciding what to believe.

            In reality, the coin is heads and the sealed card is plain.
            """),
    ),
}
