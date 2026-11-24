package infrastructure


fun traverseTree(rootNode: Node, inputs:List<Any>):Node{
    //check if current node is empty

    if not rootNode:
        return ""

    //check if current node is leaf node
    else if not rootNode.children:
        return rootNode.question

    //find if the solution is true or false at that node
    isTrue=infrastructure.decision_tree_traversal.matchInput(rootNode,inputs)

    //retrieve the current node details
    question=rootNode.text
    val (left,right)=rootNode.children

    //proceed deeper into the tree
    if not isTrue:
        return traverseTree(left,inputs)
    else:
        return traverseTree(right,inputs)

}
