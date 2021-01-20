# find pair sum values in array target to k- return the indices
'''
test cases
1. solution exist-best case
2. no solution- what to return
3. emty array
4. only one value in array
5. only one value in array and target k equal to value
6. only two value in array and solution exist

solution- start with best case if solution exist
dict{numberToFind: currentIndex}-numberToFind= target-currentIndexVal

'''
def pairSumIndices(A,k){
	aMap={}
	for i in range(len(A)):
		if(A[i]==aMap[A[i]])
			return []
}

# return the max area of water hold-array values represents height of a vertical line
'''
area= min(A[i],A[j]) * (j-i), move min(A[i],A[j]) value
does a higher line inside our container affect area-
no, lines inside a container doesn't affect the area

'''

# return the length of longest substring without repeating characters
'''
dict{currentVal:currentIndex}-sliding window using i,j pointers
'''

# breadth first search & level order of binary tree
'''
for level order- queue length is the level and increment count-length=q.length, count=0, level=[]
bfs(root)
	res=[], q=[root]
	while(q.length)
		node=q.pull;
		res.add(node.val);
		if(node.left)q.push(node.left);
		if(node.right)q.push(node.right);
	return res;
levelOrder(root)
	res=[], q=[root]
	while(q.length)
		length=q.length, count=0, level=[];
		while(count< length)
			node=q.pull;
			level.add(node.val);
			if(node.left)q.push(node.left);
			if(node.right)q.push(node.right);
			count++;
		res.add(level);
	return res;
'''

# depth first search of binary tree
'''
        		1
		2				3
	4		5				6
	  7
	8
nrl preoder-   [1, 3, 6, 2, 5, 4, 7, 8]
rnl inorder-   [6, 3, 1, 5, 2, 7, 8, 4]
rln postorder- [6, 3, 5, 8, 7, 4, 2, 1]

dfs(node, level, res)
	if(node==null) return;
	if(level>= res.length) res.add(node.val)
	if(node.right) dfs(node.right, level+1, res)
	if(node.left) dfs(node.left, level+1, res)


right side view of binary tree- [1, 3, 6, 7, 8]- level >= result array length then add node to result array
1. prioritize right side- nrl preorder
2. keep track level of nodes
rightSideView(root)
	res=[]
	dfs(root, level=0, res)
	return res

is valid binary search tree
isValidBST(root)
	if(root==null) return true
	return dfs(root, -INF, +INF)

dfs(node, min, max)
	if(node.val <= min || node.val >= max) return false;
	if(node.left) if(!dfs(node.left, min, node.val)) return false;
	if(node.right) if(!dfs(node.right, node.val, max)) return false;
	return true;
'''

# dfs and bfs of 2d array or matrix- up [-1,0], right [0,1], down [1,0], left [0,-1]
'''
[1,   2,  3,  4,  5]
[6,   7,  8,  9, 10]
[11, 12, 13, 14, 15]
[16, 17, 18, 19, 20]
dfs-[1, 2, 3, 4, 5, 10, 15, 20, 19, 14, 9, 8, 13, 18, 17, 12, 7, 6, 11, 16]
bfs-[1, 2, 6, 3, 7, 11, 4, 8, 12, 16, 5, 9, 13, 17, 10, 14, 18, 15, 19, 20]
directions= [[-1,0],[0,1],[1,0],[0,-1]]
dfs(matrix, row, col, seen, values)
	if(row<0 || col<0 || row>matrix.length || col>matrix[0].length || seen[row][col]) return;
	values.add(matrix[row][col])
	seen[row][col]= true
	for i in range(directions.length)
		currentDir= directions[i]
		dfs(matrix, row+currentDir[0], col+currentDir[1], seen, values)
values=[], seen=[][]
dfs(matrix,row=0, col=0, values)
return values

WALL=-1, GATE=0, EMPTY=INF, directions= [[-1,0],[0,1],[1,0],[0,-1]]
sequential order to look for gates, for each gate apply dfs or bfs to update distance in EMPTY cell
gateMinDistance(matrix)
	for row in range(matrix.length)
		for col in range(matrix[0].length)
			if(matrix[row][col]==GATE) dfs(matrix, row, col, currentStep=0)
	return matrix
dfs(matrix, row, col, currentStep)
	if(row<0 || row>=matrix.length || col<0 || col>=matrix[0].length || currentStep>matrix[row][col]) return;
	matrix[row][col]=currentStep;
	for i in range(directions.length)
		currentDir= directions[i];
		dfs(matrix, row+currentDir[0], col+currentDir[1], currentStep+1)


bfs(matrix)
	values=[], seen=[][], q=[[0,0]]
	while(q.length)
		currentPos= q.pull;
		row=currentPos[0], col= currentPos[1];
		if(row<0 || row>=matrix.length || col<0 || col>=matrix[0].length || seen[row][col]) continue;
		seen[row][col]=true;
		values.push[matrix[row][col]];
		for i in range(directions.length)
			currentDir=directions[i]
			q.push([row + currentDir[0], col+ currentDir[1]])
	return values

noOfIslands(matrix)
	islandCount=0, q=[];
	for row in range(matrix.length)
		for col in range(matrix[0].length)
			if(matrix[row][col]==1)
				islandCount++
				q.push([row, col])
				while(q.length)
					currentPos=q.pull, currentRow=currentPos[0], currentCol=currentPos[1];
					matrix[currentRow][currentCol]=0
					for i in range(directions.length)
						currentDir=directions[i], nextRow=currentRow+currentDir[0],
						nextCol=currentCol+currentDir[1];
						if(nextRow<0 || nextRow>=matrix.length || nextCol<0 || nextCol>=matrix[0].length)continue;
						if(matrix[nextRow][nextCol]==1) q.push([nextRow,nextCol])
	return islandCount

sequential order-count fresh oranges, put rotten oranges into queue
bfs-use queue size to track minutes, process rotting oranges-rot fresh oranges push into queue then decrement
fresh oranges count	
directions=[[-1,0], [0,1], [1,0], [0,-1]]
ROTTEN=2, FRESH=1, EMPTY=0
rottingTime(matrix)
	q=[], freshOranges=0
	for row in range(matrix.length)
		for col in range(matrix[0].length)
			if(matrix[row][col]==ROTTEN) q.push([row,col])
			if(matrix[row][col]==FRESH) freshOranges++
	currentQueueSize= q.length, minutes=0
	while(q.length>0)
		if(currentQueueSize==0)minutes++, currentQueueSize= q.length
		currentOrange=q.pull, currentQueueSize--
		row= currentOrange[0], col=currentOrange[1]
		for i in range(directions.length)
			currentDir= directions[i], nextRow=currentDir[0]+row, nextCol=currentDir[1]+col
			if(nextRow<0 || nextRow>=matrix.length || nextCol<0 || nextCol>=matrix.length) continue;
			if(matrix[nextRow][nextCol]==FRESH)
				matrix[nextRow][nextCol]=2, freshOranges--, q.push([nextRow][nextCol])
	if(freshOranges>0) return -1
	return minutes

'''

# dfs and bfs of graph- adjacency list or adjacency matrix representation
'''
bfs adjacency list graph
adjacency list=[[1,3], [0], [3,8], [0,4,5,2], [3,6], [3], [4,7], [6], [2]]
bfs=[0, 1, 3, 4, 5, 2, 6, 8, 7]
bfs(graph)
	q=[0], values=[], seen=[]
	while(q.length)
		vertex=q.pull, values.add(vertex), seen[vertex]=true
		connections= graph[vertex]
		for i in range(connections.length)
			connection= connections[i]
			if(!seen[connection]) q.push(connection)

dfs(vertex,graph,values,seen)
	values.add(vertex), seen[vertex]=true, connections=graph[vertex]
	for i in range(connections.length)
		connection= connections[i]
		if(!seen[connection]) dfs(connection, graph, values, seen)

headId=4, managers=[2,2,4,6,-1,4,4,5], managers index is employee, informTime[0,0,4,0,7,3,6,0]
adjacency list- index represent employee id and array contains its direct subordinates
graph=[[],[],[0,1],[],[2,5,6],[7],[3],[]]
timeToInform(headId, managers, informTime)
	graph=[[]]
	for i in range(managers.length)
		manager=managers[i]
		if(manager==-1) continue;
		graph[manager].add(i);
	return dfs(headId, graph, informTime)
dfs(id, graph, informTime)
	if(graph[id].length==0) return 0;
	max=0, subordinates= graph[id]
	for i in range(subordinates.length)
		max= Math.max(max, dfs(subordinates[i], graph, informTime))
	return max+ informTime[id]

n=6, prerequisite=[[1,0],[2,1],[2,5],[0,3],[4,3],[3,5],[4,5]] course 0 needs to done before course 1
graph=[[1],[2],[],[0,4],[],[2,3,4]]
inDegree=[1,1,2,1,2,0]
canFinish(n, prerequisite)
	graph=[[]], inDegree=[]
	for i in range(prerequisite.length)
		pair=prerequisite[i], graph[pair[1]].add(pair[0]), inDegree[pair[0]]++;
	stack=[]
	for i in range(inDegree.length)
		if(inDegree[i]==0) stack.push(i)
	count=0
	while(stack.length)
		current= stack.pop, count++, adjacent=graph[current];
		for i in range(adjacent.length)
			next=adjacent[i], inDegree[next]--;
			if(inDegree[next]==0) stack.push;
	return count==n


'''