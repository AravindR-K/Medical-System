


#Assume we have ocr output with tables as dataframes, images and analyses provided as text
#Assume we have range of permissible values as well
#we need a very optimal way for
#   ->using laya and medgemma only when required (range of values not provided)
#   ->providing only the necessary input at each level of the tree

matchInput(rootNode,inputs)

data class Node(
    val text: String,
val children: List<node>
)