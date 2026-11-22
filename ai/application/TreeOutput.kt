package ai.application
package ai.domain.DecisionTree

//given the root node and set of inputs,
fun treeOutput(rootNode: Node, inputs:List<Any>):Node{
    result = DecisionTree.traverseTree(rootNode,inputs)
    return result
}