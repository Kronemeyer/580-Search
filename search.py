# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack
from util import Queue
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    All other searches are a copy/paste of depthFirst using different data structures to hold 
    the fringe. In general, the problem consists of visited nodes, fringe nodes, and the
    directions to get to any nodes in the data structure. 

    Initial node is the root (starting position of pacman). This is pushed onto the fringe
    with an empty direction set. Once on the fringe, it is popped into its separate components:
    position coordinates and direction set to current node. The position coordinates are checked 
    to see whether pacman is in a goal state. If so it returns the corresponding directions set.
    Otherwise, it will use the getSuccessors function of the current node to find all possible
    children of that node and pass the directions it currently has + the new direction to the new
    node. This process is repeated until a goal state is found. 
    """
    fringe = Stack()
    dirSet,visited = [],[]
    root = problem.getStartState()
    fringe.push((root,dirSet))

    while not fringe.isEmpty():
        child, direction = fringe.pop()
        if child not in visited:
            visited.append(child)
            if problem.isGoalState(child):
                return direction
            for nextChild in problem.getSuccessors(child):
                coor = nextChild[0]
                dire = nextChild[1]
                newDirList = []
                for dirr in direction:
                    newDirList.append(dirr)
                newDirList.append(dire)
                fringe.push((coor,newDirList))
                # appendDirection = direction + [dire]
                # fringe.push((coor,appendDirection))
    return dirSet

def breadthFirstSearch(problem):
    fringe = Queue()
    dirSet, visited = [],[]
    root = problem.getStartState()
    fringe.push((root,dirSet))

    while not fringe.isEmpty():
        child, direction = fringe.pop()
        if child not in visited:
            visited.append(child)
            if problem.isGoalState(child):
                return direction
            for nextChild in problem.getSuccessors(child):
                coor = nextChild[0]
                dire = nextChild[1]
                newDirList = []
                for dirr in direction:
                    newDirList.append(dirr)
                newDirList.append(dire)
                fringe.push((coor,newDirList))
                # appendDirection = direction + [dire]
                # fringe.push((coor,appendDirection))
    return dirSet

def uniformCostSearch(problem):
    """
    Uniform Cost Search is simply A* search with a null heuristic
    """
    return aStarSearch(problem)   

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    fringe = PriorityQueue()
    dirSet, visited = [],[]
    root = problem.getStartState()
    fringe.push((root,dirSet),0)
    while fringe:
        child, direction = fringe.pop()
        if child not in visited:
            visited.append(child)
            if problem.isGoalState(child):
                return direction
            for nextChild in problem.getSuccessors(child):
                coor = nextChild[0] 
                dire = nextChild[1]
                cost = nextChild[2]
                newDirList = []
                for dirr in direction:
                    newDirList.append(dirr)
                newDirList.append(dire)
                costPlusHeuristic = problem.getCostOfActions(newDirList) + heuristic(coor, problem)
                fringe.push((coor,newDirList),costPlusHeuristic)
    return dirSet


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
