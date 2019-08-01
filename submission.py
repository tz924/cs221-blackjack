import util, math, random
from collections import defaultdict, Counter
from util import ValueIteration


############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a
# counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if
        # you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if
        # you deviate from this)
        return ["-1", "+1"]
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward)
    # tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty
    # list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if
        # you deviate from this)
        return [(-1, .999, 1), (1, .001, 2019)] if state == 0 else list()
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if
        # you deviate from this)
        return 0.99
        # END_YOUR_CODE


############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in
        the deck)
        multiplicity: single integer representing the number of cards with
        each face value
        threshold: maximum number of points (i.e. sum of card values in hand)
        before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation
    # for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the
    #   player's hand.
    #   -- If the player's last action was to peek, the second element is the
    #   index
    #      (not the face value) of the next card that will be drawn;
    #      otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards
    #   remaining
    #      in the deck, or None if the deck is empty or the game is over (
    #      e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the
    # succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward)
    # tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of
    # cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry
        # if you deviate from this)
        totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts = state

        # end state
        if not deckCardCounts:
            return []

        # terminal state
        if totalCardValueInHand > self.threshold:  # busting
            return [((totalCardValueInHand, None, None), 1, 0)]

        def take(index):
            countAfter = deckCardCounts[index] - 1
            if countAfter < 0:
                return None
            cardValue = self.cardValues[index]
            newCounts = deckCardCounts[:index] \
                        + tuple([countAfter]) \
                        + deckCardCounts[index + 1:]
            return cardValue, newCounts

        if action[0] == 'Q':  # quitting
            return [((totalCardValueInHand, None, None), 1,
                     totalCardValueInHand)]

        if action[0] == 'T':
            # If last action is peek, taken peeked card
            if nextCardIndexIfPeeked:
                cardValue, newCounts = take(nextCardIndexIfPeeked)
                newTotal = totalCardValueInHand + cardValue
                newState = (newTotal, None, newCounts) \
                    if newTotal <= self.threshold \
                    else (newTotal, None, None)

                # Take cause empty
                if not any(newCounts):
                    return [((totalCardValueInHand + cardValue, None, None),
                            1, totalCardValueInHand + cardValue)]

                # Deterministic => p = 1
                return [(newState, 1, 0)]

            # Otherwise return all possible
            else:
                newStates = []
                for i, _ in enumerate(deckCardCounts):
                    taken = take(i)
                    if not taken:
                        continue
                    cardValue, newCounts = taken
                    newTotal = totalCardValueInHand + cardValue
                    newState = (newTotal, None, newCounts) \
                        if newTotal <= self.threshold \
                        else (newTotal, None, None)
                    newStates.append(newState)

                    # Take cause empty
                    if not any(newCounts):
                        return [((totalCardValueInHand + cardValue, None, None),
                                1, totalCardValueInHand + cardValue)]

                prob = 1. / len(newStates)
                return [(s, prob, 0) for s in newStates]

        elif action[0] == 'P':
            # Peek twice in a row
            if nextCardIndexIfPeeked:
                return []

            newStates = []
            for i, c in enumerate(deckCardCounts):
                if c == 0:
                    continue

                newState = (totalCardValueInHand, i, deckCardCounts)
                newStates.append(newState)

            return [(s, 1. / len(newStates), -self.peekCost)
                        for s in newStates]
        # END_YOUR_CODE

    def discount(self):
        return 1


############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if
    # you deviate from this)
    return BlackjackMDP(cardValues=[2, 3, 4, 5, 16, 17, 18, 19], multiplicity=3,
                        threshold=20, peekCost=1)
    # END_YOUR_CODE


############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a
# list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor,
                 explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in
                       self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to
    # update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to
    # check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry
        # if you deviate from this)
        # Terminal state
        if not newState:
            return
        eta = self.getStepSize()
        prediction = self.getQ(state, action)
        Vopt = max(self.getQ(newState, newAction)
                   for newAction in self.actions(newState))
        target = reward + self.discount * Vopt
        # Only phi is different so it goes into the loop
        for f, phi in self.featureExtractor(state, action):
            gradient = (prediction - target) * phi
            self.weights[f] -= eta * gradient
        # END_YOUR_CODE


# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10,
                        peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                        threshold=40, peekCost=1)


def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it
    # will probably be useful
    # to you as you work to answer question 4b (a written question on this
    # assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate
    # Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two
    # approaches.
    # BEGIN_YOUR_CODE
    mdp.computeStates()
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
                            featureExtractor, 0.2)
    util.simulate(mdp, rl, 30000)
    mdp.explorationProb = 0
    qDict = {s: rl.getAction(s) for s in mdp.states}

    vi = ValueIteration()
    vi.solve(mdp)

    logs = []
    for s, a in vi.pi.items():
        if a == qDict[s]:
            logs.append(0)
        else:
            logs.append(1)
    print("{} total, {} same, {} different"
          .format(len(logs), len(logs) - sum(logs), sum(logs)))
    print("{}% diff".format(100. * sum(logs) / len(logs)))
    print
    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in
# the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the
#       presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each
# face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if
    # you deviate from this)
    features = []

    # -- Indicator for the action and the current total (1 feature).
    features += [((action, total), 1)]

    if counts:
        # -- Indicator for the action and the presence/absence of each face
        # value in the deck.
        features += [((action, tuple(1 if e else 0 for e in counts)), 1)]

        # -- Indicators for the action and the number of cards remaining with
        # each face value (len(counts) features).
        for i, c in enumerate(counts):
            features += [((action, i, c), 1)]

    return features
    # END_YOUR_CODE


############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10,
                           peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15,
                               peekCost=1)


def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely
    # optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a
    # written question).
    # Consider adding some code here to simulate two different policies over
    # the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE

    # First, run value iteration on the originalMDP to compute an optimal
    # policy for that MDP.

    TRIALS = 30000
    expected = lambda l: 1. * sum(l) / len(l)
    # Next, simulate your policy on newThresholdMDP by calling simulate with
    # an instance of FixedRLAlgorithm that has been instantiated using the
    # policy you computed with value iteration. What is the expected reward
    # from this simulation?
    original_mdp.computeStates()
    vi = ValueIteration()
    vi.solve(original_mdp)
    # rl = util.FixedRLAlgorithm(defaultdict(lambda: "Take", vi.pi))
    modified_mdp.computeStates()
    rl = util.FixedRLAlgorithm(vi.pi)

    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    totalRewards = []  # The rewards we get on each trial
    for trial in range(TRIALS):
        state = modified_mdp.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for _ in range(1000):
            try:
                action = rl.getAction(state)
            except KeyError:
                continue
            transitions = modified_mdp.succAndProbReward(state, action)
            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= modified_mdp.discount()
            state = newState
        totalRewards.append(totalReward)
    print("VI then Fixed: {}".format(expected(totalRewards)))

    # Now try simulating Q-learning on originalMDP (30,000 trials).
    rl = QLearningAlgorithm(original_mdp.actions, original_mdp.discount(),
                            featureExtractor)
    util.simulate(original_mdp, rl, TRIALS)

    # Then, using the learned parameters, run Q-learning again on
    # newThresholdMDP (again, 30000 trials).
    rewards = util.simulate(modified_mdp, rl, TRIALS)
    print("Double Q-learning: {}".format(expected(rewards)))

    # What is your expected reward under the new Q-learning policy? Provide
    # some explanation for how the rewards compare with when FixedRLAlgorithm
    # is used. Why they are different?
    # END_YOUR_CODE
