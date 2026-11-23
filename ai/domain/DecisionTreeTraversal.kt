package infrastructure


fun traverseTree(rootNode: Node, inputs:List<Any>):Node{
    //check if current node is empty
    if not rootNode:
        return

    //check if current node is leaf node
    else if not rootNode.children:
        return rootNode.question

    //find the input that can solve the question in the node
    matchingInput=infrastructure.matchInput(rootNode,inputs)

    //retrieve the current node details
    question=rootNode.text
    val (left,right)=rootNode.children

    //find if the node has been solved
    isTrue=infrastructure.solveQuestion(question,matchingInput)

    //proceed deeper into the tree
    if not isTrue:
        return traverseTree(left,inputs)
    else:
        return traverseTree(right,inputs)

    //return the result
    return result
}