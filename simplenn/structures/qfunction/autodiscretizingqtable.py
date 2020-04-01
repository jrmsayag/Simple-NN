import numpy as np
import random as rd

from .. import qfunction

class DecisionNode:
    """
    The core component of discretization decision-tree that is
    automatically built by an AutoDiscretizingQTable.
    """

    def __init__(self, groupId, maxObs):
        """
        Constructor.
        """

        self.groupId = groupId
        self.maxObs = maxObs

        self.splitIdx = None
        self.splitVal = None

        self.nextNode1 = None
        self.nextNode2 = None

        self.nObs = 0

        self.sumState = None

    def copy(self):
        """
        Returns an exact copy of the (sub) decision tree starting from this node.
        """

        newNode = DecisionNode(self.groupId, self.maxObs)

        newNode.splitIdx = self.splitIdx
        newNode.splitVal = self.splitVal

        newNode.nObs = self.nObs

        if self.sumState is not None:
            newNode.sumState = self.sumState.copy()

        if self.nextNode1 is not None:
            newNode.nextNode1 = self.nextNode1.copy()
        if self.nextNode2 is not None:
            newNode.nextNode2 = self.nextNode2.copy()

        return newNode

    def buildSplitIdxAndVal(self, s):
        """
        Decides which index of the input state (and the corresponding
        value) this node should be split on.
        """

        # Choosing the split index
        maxIdx = len(self.sumState) - 1
        self.splitIdx = rd.randint(0, maxIdx)

        # Choosing the split value
        self.splitVal = self.sumState[self.splitIdx] / self.nObs

    def split(self, s, nextGroupId):
        """
        Effectively split this node by creating two children leaf nodes.
        """

        self.buildSplitIdxAndVal(s)

        self.nextNode1 = DecisionNode(self.groupId, self.maxObs)
        self.nextNode2 = DecisionNode(nextGroupId, self.maxObs)

        return (self.nextNode1, self.nextNode2)

    def shouldSplit(self):
        """
        Indicates whether this node should be split.
        """

        return self.nObs >= self.maxObs

    def observe(self, s):
        """
        Updates the statistics of this node with the given observed state.
        """

        if self.nObs:
            self.sumState += s
        else:
            self.sumState = s.copy()

        self.nObs += 1

class AutoDiscretizingQTable(qfunction.QTable):
    """
    A Q-Table whose states are dynamically estimated from a continuous
    observation space as learning proceeds.
    """

    MODE_0 = 0
    MODE_SPLIT = 1

    def __init__(
        self, actions, gamma, alphaRate, alphaCutoff, epsInit, epsFinal, epsRampLen,
        defaultVal, maxObsPerLeaf, maxLeafNodes, splitResetMode
    ):
        """
        Constructor.
        """

        super().__init__(
            actions, gamma, alphaRate, alphaCutoff, epsInit, epsFinal, epsRampLen,
            defaultVal
        )

        self.maxLeafNodes = maxLeafNodes
        self.maxObsPerLeaf = maxObsPerLeaf

        self.splitResetMode = splitResetMode

        self.rootNode = DecisionNode(0, maxObsPerLeaf)
        self.nextGroupId = 1

    def copy(self):
        """
        Returns an exact copy of this Q-function.
        """

        newQ = AutoDiscretizingQTable(
            self.actions,
            self.gamma,
            self.alphaRate,
            self.alphaCutoff,
            self.epsInit,
            self.epsFinal,
            self.epsRampLen,
            self.defaultVal,
            self.maxObsPerLeaf,
            self.maxLeafNodes,
            self.splitResetMode
        )

        newQ.data = self.copyData()
        newQ.isLearning = self.isLearning

        newQ.rootNode = self.rootNode.copy()
        newQ.nextGroupId = self.nextGroupId

        return newQ

    def getLeafNode(self, s):
        """
        Applies the decision tree.
        """

        nextNode = self.rootNode

        while nextNode.splitIdx is not None:

            if s[nextNode.splitIdx] <= nextNode.splitVal:
                nextNode = nextNode.nextNode1
            else:
                nextNode = nextNode.nextNode2

        return nextNode

    def getData(self, s, a=None):
        """
        Wrapper that calls the parent implementation with the discretized state.
        """

        node = self.getLeafNode(s)

        return super().getData(node.groupId, a)

    def setData(self, s, a, aData):
        """
        Wrapper that calls the parent implementation with the discretized state.

        If needed the discretizing decision tree is updated.
        """

        node = self.getLeafNode(s)

        super().setData(node.groupId, a, aData)

        node.observe(s)

        if self.nextGroupId < self.maxLeafNodes and node.shouldSplit():

            node.split(s, self.nextGroupId)

            sData = self.data[node.groupId]
            newSData = {}

            for a, aData in sData.items():

                if self.splitResetMode == self.MODE_0:
                    aData[self.N_IDX] = 0
                else:
                    aData[self.N_IDX] /= 2

                newSData[a] = {
                    self.N_IDX: aData[self.N_IDX],
                    self.Q_IDX: aData[self.Q_IDX]
                }

            self.data[self.nextGroupId] = newSData

            # One of the new nodes is given the groupId of its parent
            # (which is ok because the parent is no more a leaf node)
            # so we only add 1 to the nextGroupId.
            self.nextGroupId += 1
