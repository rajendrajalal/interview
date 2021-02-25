import java.util.*;
import java.util.stream.Collectors;

public class Codejava {

	static int[] pairSumIndicesTargetK(int[] A, int k){
		Map<Integer, Integer> map = new HashMap();
		int[] pairArray = new int[2];
		int numberToFind=0;
		for ( int i=0; i< A.length; i++ ) {
			numberToFind = k-A[i];
			if(map.containsKey(A[i])){
				pairArray[0]=map.get(A[i]);
				pairArray[1]=i;
				return pairArray;
			}
			map.put(numberToFind, i);				
		}
        return pairArray;	
	}

	static int maxAreaWaterContainer(int[] A){
		int i =0, j=A.length-1, maxArea=0;
		while(i<j){
			maxArea= Math.max(Math.min(A[i], A[j]) * (j-i),maxArea);
			if(A[i] < A[j]) i++;
			else j--;
		}
		return maxArea;
	}

	static int totalTrappedWater(int[] A){
		int maxLeft=0, maxRight=0, currentIndexTrappedWater=0, totalTrappedWater=0;		
		for(int i=0; i< A.length; i++){
			int left = i, right= i;
            maxLeft=0; 
            maxRight=0;			
			while(left>=0){
				maxLeft= Math.max(maxLeft, A[left]);
				left--;
			}
			while(right<A.length){
				maxRight= Math.max(maxRight, A[right]);
				right++;
			}
			currentIndexTrappedWater = Math.min(maxLeft, maxRight) - A[i];
			if (currentIndexTrappedWater>0) {
				totalTrappedWater+= currentIndexTrappedWater;
			}
		}
		return totalTrappedWater;
    }

    static int total_TrappedWater(int[] A){
    	int left=0, right=A.length-1, maxLeft=0, maxRight=0, currentIndexTrappedWater=0,                 totalTrappedWater=0;
    	while(left < right){
    		maxLeft = Math.max(maxLeft, A[left]);
    		maxRight = Math.max(maxRight, A[right]);
    		if (maxLeft < maxRight) {
    			currentIndexTrappedWater= maxLeft-A[left];    			
    			left++;
    		}else{
    			currentIndexTrappedWater= maxRight - A[right];    			
    			right--;
    		}
            if (currentIndexTrappedWater>0) {
    				totalTrappedWater+= currentIndexTrappedWater;
    			}
    	}
    	return totalTrappedWater;
    }

    

    static int longestSubstringLength(String s){
    	char[] A = s.toCharArray();
    	int start=0, maxLength=0;
    	Map<Character, Integer> map = new HashMap();    	
    	for(int end =0; end < A.length; end++){    		
    		if (map.containsKey(A[end]) && (map.get(A[end])>=start)) {
    			start= map.get(A[end])+1;    			
    		}
    		map.put(A[end], end);    		
    		maxLength = Math.max(maxLength, end-start+1);            
    	}
    	return maxLength;    
    }

   public static class ListNode {
      int val;
      ListNode next;
      ListNode prev;
      ListNode child;
      ListNode() {}
      ListNode(int val) { this.val = val; }
      ListNode(int val, ListNode next, ListNode prev, ListNode child) { 
        this.val = val; this.next = next; this.prev = prev; this.child = child;}
  }

  static ListNode reverseLinkedList(ListNode head){
      ListNode currentNode = head;
      ListNode reverseLinkedListHead = null;
      while(currentNode!=null){
        ListNode nextNode = currentNode.next;
        currentNode.next = reverseLinkedListHead;
        reverseLinkedListHead = currentNode;
        currentNode = nextNode;
      }
      return reverseLinkedListHead;
  }
  static ListNode reverseLinkedListPositionMN(ListNode head, int M, int N){//index start from 1,track m-1,m,n,n+1
      int currentPosition = 1;
      ListNode currentNode = head, nodeBeforeM =null,reverseLinkedListMNHead =null, nodeAtMTail = null;
      while(currentPosition <M){
          nodeBeforeM = currentNode;
          currentNode = currentNode.next;
          currentPosition++;
      }
      nodeAtMTail = currentNode;
      while(currentPosition >=M && currentPosition <=N){//reverse Linked List from M to N both inclusive
          ListNode nextNode = currentNode.next;
          currentNode.next = reverseLinkedListMNHead;
          reverseLinkedListMNHead = currentNode;
          currentNode = nextNode;
          currentPosition++;
      }      
      nodeAtMTail.next = currentNode;
      if(M==1) return reverseLinkedListMNHead;
      else{
          nodeBeforeM.next = reverseLinkedListMNHead;
          return head;
      }      
  }
  static ListNode flattenDoublyLinkedList(ListNode head){// top down approach
      ListNode currentNode = head, childNodeTail = null;
      while(currentNode!=null){
        if(currentNode.child !=null){
          childNodeTail = currentNode.child;
          while(childNodeTail.next!=null){
            childNodeTail = childNodeTail.next;            
          } 
          childNodeTail.next = currentNode.next;
          if(childNodeTail.next!=null)currentNode.next.prev = childNodeTail;//currentNode could be last index
          currentNode.next = currentNode.child;
          currentNode.child.prev = currentNode;// or currentNode.next.prev = currentNode;
          currentNode.child = null;                 
        }else{
          currentNode = currentNode.next;
        }       
      }
    return head;
  }
  static ListNode findCyclicNodeLinkedList(ListNode head){     
     ListNode slow = head, fast = head, startNode = head;
     while(fast!=null && fast.next!=null){ // while(true){ check if fast & fast.next null condition inside}    
      slow = slow.next;
      fast = fast.next.next;
      if(slow == fast){        
        while(slow != startNode){
          slow = slow.next;
          startNode = startNode.next;
        }
      return startNode;
      }      
     }
     return null;
  }
public static class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode() {}
      TreeNode(int val) { this.val = val; }
      TreeNode(int val, TreeNode left, TreeNode right) {
          this.val = val;
          this.left = left;
          this.right = right;
      }      
	  }

    static TreeNode buildTreeLevelOrder(int[] A){
       Queue<TreeNode> q = new ArrayDeque();
       TreeNode root = new TreeNode(A[0]);
       TreeNode currentNode = null;
       q.offer(root);       
       for (int i=1; i< A.length ; i++ ) {
         TreeNode node = null;
         if (A[i]>0) {
           node = new TreeNode(A[i]);
           q.offer(node);
         }   
         if (i%2 !=0) {
             currentNode = q.poll();         
            currentNode.left = node;            
         }else{
            currentNode.right = node;            
         }
       }
       return root;
    }

  static List<Integer> bfsBinaryTree(TreeNode root){
      List<Integer> bfsList = new ArrayList();
      if (root == null) return bfsList;     
      Queue<TreeNode> q = new ArrayDeque();
      q.offer(root);
      while(!q.isEmpty()){
          TreeNode currentNode = q.poll();
          bfsList.add(currentNode.val);
          if(currentNode.left!=null) q.offer(currentNode.left);
          if(currentNode.right!=null) q.offer(currentNode.right);
      }
      return bfsList;
  }

  
	static List<List<Integer>> levelOrderBinaryTree(TreeNode root){
		   List<List<Integer>> levelOrderList = new ArrayList<>();        
        if(root==null) return levelOrderList;
        Queue<TreeNode> q = new ArrayDeque();
        q.offer(root);
        while (!q.isEmpty()){
            int currentLevelLength = q.size(), count = 0;
            List<Integer> currentLevelList = new ArrayList();
            while (count < currentLevelLength){             
                    TreeNode currentNode = q.poll();
                    currentLevelList.add(currentNode.val);
                    if (currentNode.left!=null) {
                        q.offer(currentNode.left);          
                    }
                    if (currentNode.right!=null) {
                        q.offer(currentNode.right);
                    }
                    count++;                 
            }
            levelOrderList.add(currentLevelList);         
        }
    return levelOrderList;
	}

  static List<Integer> dfsBinaryTree(TreeNode node){
      List<Integer> dfsList = new ArrayList();
      dfsList = dfsHelper(node, dfsList);
      return dfsList;
  }
  static List<Integer> dfsHelper(TreeNode node, List<Integer> dfsList) {
        if (node == null) return dfsList;
        dfsHelper(node.left, dfsList);
        dfsList.add(node.val);
        dfsHelper(node.right, dfsList);        
        return dfsList;
  }
  static List<Integer> dfsMatrix(int[][] maxtrix){
    boolean[][] seen = new boolean[maxtrix.length][maxtrix[0].length];
    List<Integer> dfsList = new ArrayList<>();
    dfsList = dfsMatrixHelper(maxtrix, 0, 0, seen, dfsList);// 0,0- represents starting point
    return dfsList;
  }
  static List<Integer> dfsMatrixHelper(int[][] matrix, int row, int col, boolean[][] seen, List<Integer> dfsList){
    int[][] directions = {{-1,0},{0,1},{1,0},{0,-1},};
    if(row<0 || row>= matrix.length || col <0 || col>=matrix[0].length || seen[row][col]) return dfsList;
    dfsList.add(matrix[row][col]);
    seen[row][col] = true;
    for(int[] currentDirection :directions){      
      dfsMatrixHelper(matrix, row+ currentDirection[0], col+currentDirection[1], seen, dfsList);
    }
    return dfsList;
  }
  static int dfsIslandCount(int[][] matrix){
     int islandCount =0;
     for(int row =0; row< matrix.length; row++){
        for(int col=0; col< matrix[0].length; col++){
            if(matrix[row][col] ==1){
              islandCount++;
              dfsIslandCountHelper(matrix, row, col);
            }
        }
     }
     return islandCount++;
  }
  static void dfsIslandCountHelper(int[][] matrix, int row, int col){
          int[][] directions = {{-1,0},{0,1},{1,0},{0,-1},};
          if(row<0 || row>= matrix.length || 
             col <0 || col>=matrix[0].length || matrix[row][col]==0) return;          
          matrix[row][col]=0;
          for(int[] currentDirection: directions){                
                int currentRow = row + currentDirection[0];
                int currentCol = col + currentDirection[1];
                dfsIslandCountHelper(matrix, currentRow, currentCol);
          }             
  }
  static int[][] dfsMinDistanceWallGate(int[][] matrix){
   for(int row=0; row< matrix.length; row++){
    for(int col=0; col< matrix[0].length; col++){
        if(matrix[row][col]==0) dfsMinDistanceWallGateHelper(matrix, row, col, 0);
    }
   }
   return matrix;
  }
  static void dfsMinDistanceWallGateHelper(int[][] matrix,int row, int col, int level){// current level or step
     final int WALL =-1, GATE =0, EMPTY = 100;
     int[][] directions = {{-1,0},{0,1},{1,0},{0,-1},};// up, right, down, left
     if(row<0 || row>=matrix.length || col <0 || col>=matrix[0].length// adding matrix[row][col]== GATE changes ans
        || matrix[row][col]== WALL || level > matrix[row][col]) return;// it can enter to starting gate also
     matrix[row][col] = level; 
     for(int[] currentDirection : directions){              
              int nextRow = currentDirection[0] + row;
              int nextCol = currentDirection[1] + col;
              dfsMinDistanceWallGateHelper(matrix, nextRow, nextCol, level+1);      
      }      
    }
  static int[][] bfsMinDistanceWallGate(int[][] matrix){     
     Queue<int[]> q = new ArrayDeque<>();
     for(int row =0; row< matrix.length; row++){
        for(int col=0; col< matrix[0].length; col++){
            if(matrix[row][col]==0) q.offer(new int[]{row,col});
        }
     }
     bfsMinDistanceWallGateHelper(matrix, q);
     return matrix;
  }
  static void bfsMinDistanceWallGateHelper(int[][] matrix,Queue<int[]> q){
     final int WALL =-1, GATE =0, EMPTY = 100;
     int[][] directions = {{-1,0},{0,1},{1,0},{0,-1},};// up, right, down, left
     int currentLevelLength = q.size(), currentStep=1;
     while(!q.isEmpty()){
        if(currentLevelLength==0){
          currentStep++;
          currentLevelLength = q.size();
        }
        int[] currentPosition = q.poll();
        currentLevelLength--;
        int currentRow = currentPosition[0];
        int currentCol = currentPosition[1];
        for(int[] direction : directions){
            int nextRow = currentRow + direction[0];
            int nextCol = currentCol + direction[1];
            if(nextRow<0 || nextRow >= matrix.length || nextCol <0 || nextCol >= matrix[0].length 
               || matrix[nextRow][nextCol]== WALL || currentStep > matrix[nextRow][nextCol]) continue;
            q.offer(new int[]{nextRow, nextCol});
            matrix[nextRow][nextCol] = currentStep;            
        }        
     }
  }
     
  static List<Integer> bfsMatrix(int[][] matrix){//output in zigzag manner from 0,0 using current directions[][]
    int[][] directions = {{-1,0},{0,1},{1,0},{0,-1},};// up, right, down, left
    boolean[][] seen = new boolean[matrix.length][matrix[0].length];
    List<Integer> bfsList = new ArrayList<>();
    Queue<int[]> q = new ArrayDeque<>();    
    q.add(new int[]{0,0}); 
    while(!q.isEmpty()){
      int[] currentPosition = q.poll();
      int currentRow = currentPosition[0];
      int currentCol = currentPosition[1];
      if(currentRow <0 || currentRow >= matrix.length || 
         currentCol <0 || currentCol>= matrix[0].length || seen[currentRow][currentCol]) continue;
      seen[currentRow][currentCol] = true;
      bfsList.add(matrix[currentRow][currentCol]);
      for(int[] currentDirection : directions){        
        int[] nextPosition = new int[]{currentRow+ currentDirection[0], currentCol+ currentDirection[1]};
        q.offer(nextPosition);
      }
    }
    return bfsList;
  }
  static int bfsIslandCount(int[][] islandMatrix){
    int[][] directions = {{-1,0},{0,1},{1,0},{0,-1},};
    int islandCount = 0;
    Queue<int[]> q = new ArrayDeque<>();
    for(int row =0; row < islandMatrix.length; row++){
      for(int col=0; col < islandMatrix[0].length; col++){
        if(islandMatrix[row][col]==1) {
              islandCount++;
              islandMatrix[row][col] =0;
              q.offer(new int[]{row,col});
              while(!q.isEmpty()){
              int[] currentPosition = q.poll();
              int currentRow = currentPosition[0];
              int currentCol = currentPosition[1];        
              for(int[] currentDirection: directions){                  
                  int nextRow = currentRow + currentDirection[0];
                  int nextCol = currentCol + currentDirection[1];
                  if(nextRow <0 || nextRow >= islandMatrix.length ||
                      nextCol <0 || nextCol >= islandMatrix[0].length) continue;
                  if(islandMatrix[nextRow][nextCol] ==1) {
                      islandMatrix[nextRow][nextCol] =0;
                      q.offer(new int[]{nextRow,nextCol}); 
                   }
              }
           }   
        }
      }
    }
   return islandCount;
  }

  static int bfsMinutesToRotOrange(int[][] orangeMatrix){// rotten orange=2, fresh orange =1 
      final int ROTTEN = 2, FRESH = 1, EMPTY = 0;
      int rottenOranges=0, freshOranges =0, noOfMinutes =0;
      int[][] directions = {{-1,0},{0,1},{1,0},{0,-1},};// up, right, down, left    
      Queue<int[]> q = new ArrayDeque<>();
      for(int row =0; row< orangeMatrix.length; row++){
          for(int col=0; col< orangeMatrix[0].length; col++){
              if(orangeMatrix[row][col]== ROTTEN) {
                   rottenOranges++;
                   q.offer(new int[]{row,col});
                }
              if(orangeMatrix[row][col]== FRESH) freshOranges++;
          }
      }
      int currentLevelOranges = q.size();      
      while(!q.isEmpty()){
          if(currentLevelOranges ==0){
              noOfMinutes++;
              currentLevelOranges = q.size();
          }                    
          int[] currentPosition = q.poll();
          currentLevelOranges--;
          int currentRow = currentPosition[0];
          int currentCol = currentPosition[1];
          for (int[] currentDirection: directions) {              
              int nextRow = currentRow + currentDirection[0];
              int nextCol = currentCol + currentDirection[1];
              if(nextRow <0 || nextRow >= orangeMatrix.length ||  nextCol <0 ||
                 nextCol >= orangeMatrix[0].length || orangeMatrix[nextRow][nextCol]== EMPTY
                 || orangeMatrix[nextRow][nextCol] == ROTTEN) continue;                  
              orangeMatrix[nextRow][nextCol] = ROTTEN;
              freshOranges--;
              q.offer(new int[]{nextRow, nextCol});                              
              }                    
      }
      return freshOranges==0 ? noOfMinutes : -1;
  }
  static int maxDepth(TreeNode node){
      int currentDepth = 0;
      currentDepth = maxDepthHelper(node, currentDepth);
      return currentDepth;
  }
  static int maxDepthHelper(TreeNode node, int currentDepth){
      if (node == null ) return currentDepth;
      currentDepth++;
      return Math.max(maxDepthHelper(node.right, currentDepth), maxDepthHelper(node.left, currentDepth));
  }
  static List<Integer> rightSideViewBFS(TreeNode node){
      List<Integer> rightSideViewList = new ArrayList();
      if(node==null) return rightSideViewList; 
      Queue<TreeNode> q = new ArrayDeque();
      q.offer(node);
      int currentLevelLength = q.size();
      while(!q.isEmpty()){        
         if(currentLevelLength ==0) currentLevelLength = q.size();        
         TreeNode currentNode = q.poll();
         currentLevelLength--;
         if(currentLevelLength==0) rightSideViewList.add(currentNode.val);
    //  if(count == 0) rightSideViewList.add(currentNode.val);
         if (currentNode.left!=null) q.offer(currentNode.left);
         if(currentNode.right!=null) q.offer(currentNode.right);                 
       }      
      return rightSideViewList;
  }
  static List<Integer> rightSideViewDFS(TreeNode root){
    List<Integer> rightSideViewList = new ArrayList();
    int currentLevel = 0;
    rightSideViewList = rightSideViewHelper(root, currentLevel, rightSideViewList);
    return rightSideViewList;
  }
  static List<Integer> rightSideViewHelper(TreeNode node, int currentLevel, List<Integer> rightSideViewList){
      if(node==null) return rightSideViewList;
// first occurence of node at every level, check if it's size is greater than or equal to list size then add it
      if(currentLevel == rightSideViewList.size()) rightSideViewList.add(node.val);       
      if(node.right!=null) rightSideViewHelper(node.right, currentLevel+1, rightSideViewList);
      if(node.left!=null) rightSideViewHelper(node.left, currentLevel+1, rightSideViewList);
      return rightSideViewList;
  }
/*
  static int nodesInCompleteBinaryTree(TreeNode root){
    int height = getTreeHeight(root);
    int leftIndex =0, rightIndex = Math.pow(2, height-1) -1;
    int nodesExceptLastLevel = rightIndex;
    nodesAtLastLevel = binarySearchInTree(root, height, leftIndex, rightIndex);
    return nodesExceptLastLevel + nodesAtLastLevel;  
  }

  static int getTreeHeight(TreeNode root){
      int height = 0;
      while(root!=null){
        root = root.left;
        height++;
     }
     return height;
  }

  static int binarySearchInTree(TreeNode node, int height, int leftIndex, int rightIndex){
      int indexToFind = Math.ceil((leftIndex + rightIndex)/2);
      while(leftIndex< rightIndex){
        if (nodeExist(node, height, indexToFind)) leftIndex = indexToFind;
        else rightIndex = indexToFind-1;              
      }
      return leftIndex +1; // return rightIndex check for it
  }

  static boolean nodeExist(TreeNode node, int height, int indexToFind){
      int leftIndex =0, rightIndex = Math.pow(2, height-1), currentHeight =0;
      while(currentHeight < height){
        int midNode = Math.ceil((leftIndex+rightIndex)/2);
        if(indexToFind>= midNode){
          node= node.right;
          leftIndex = midNode;
        }else{
          node = node.left;
          rightIndex = midNode-1;
        }
      }
      return node!=null; 
  }
*/
  static boolean isValidBST(TreeNode root){
      if(root==null) return true;
// going left- update left node lesser than value, going right- update right node greater than value
      return isValidBSTHelper(root, Long.MIN_VALUE,Long.MAX_VALUE);
  }
  static boolean isValidBSTHelper(TreeNode node, long min, long max){    
    if(node.val <= min || node.val >= max) return false;
    if(node.left!=null){
      if(!isValidBSTHelper(node.left, min, node.val)) return false;      
    }
    if(node.right!=null){
      if(!isValidBSTHelper(node.right, node.val, max)) return false;      
    }
    return true;
  }
  /*
       3
  9         20
    5    15     7


FULL BINARY TREE - each node has two children or no children
            3
       9         20
              15     7

COMPLETE BINARY TREE- every level full except last level in sequential manner
          3
    9           20
 4     5    15

FULL AND COMPLETE BINARY TREE- every level full 
no. of nodes except last level- 2 pow (height-1) - 1 and at last level- 2 pow (height-1)
        3
   9         20
4     5   15     7

BINARY SEARCH TREE - each node value greater than the every node in left tree, ||ly lesser than right tree nodes
          12
     8          18     
  5   10     14    25

HEAP RESEMBLES COMPLETE BINARY TREE- like maxheap each node value is greater than its left child
and right child
[50, 40, 25, 20, 35, 10, 15] parent= floor((index-1)/2), leftChild = (index * 2)+1, rightChild = (index * 2)+2
PRIORITY QUEUE- size, isEmpty, peek, pop/deletion-check left & right child, push/insertion-check parent
             50
      40            25
  20      35     10     15

*/
// use case- kth largest number from array, create priority queue insert array values.
 /* public static class PriorityQueue {// maxHeap
    List<Integer> maxHeap;
    PriorityQueue(){
      this.maxHeap = new ArrayList<>();
    }
    static int _parentIndex(int index){
      return Math.floor((index-1)/2);      
    }
    static int _leftChildIndex(int index){
      return (index * 2) +1;    
    }
    static int _rightChildIndex(int index){
      return (index * 2) +2;
    }
    static void _swap(int i, int j){
      int temp = this.maxHeap.get(i);
      this.maxHeap.get(i) = this.maxHeap.get(j);
      this.maxHeap.get(j) = temp;
    }
    static boolean _compare(int i, int j){
      return this.maxHeap.get(i) > this.maxHeap.get(j);
    }
    static void _shiftUp(){
      int lastIndex = this.size()-1;     
      while(lastIndex>0 && this._compare(lastIndex, this._parentIndex(int lastIndex)){
        this._swap(lastIndex, this._parentIndex(int lastIndex));
        lastIndex = this._parentIndex(int lastIndex);
      }
    }
    static void _shiftDown(){
      int index = 0;
      while( (this._leftChildIndex(index) > this.size()  && this._compare(this._leftChildIndex(index), index)) ||
        (this._rightChildIndex(index) > this.size()  && this._compare(this._rightChildIndex(index), index))){
        greaterChildIndex = ternary operator;
        this._swap(greaterChildIndex, index);
      }
    }
    static int push(int val){
      this.maxHeap.add(val);
      this._shifUp();
    }
    static int pop(){
      this._swap(0, this.size()-1);
      int maxHeapVal = this.maxHeap.get(this.size()-1);
      this._shiftDown();
      return maxHeapVal;
    }
    static int size(){
      return this.maxHeap.length();
    }
    static int isEmpty(){
      return this.size() ==0;
    }
    static int peek(){
      return this.maxHeap.get(0);
    }
    static push(int val){

    }
  }
*/
  static void createAdjacencyListGraph(){

  }
  static void createAdjacencyMatrixGraph(){

  }
  static List<Integer> bfsAdjacencyListGraph(List<List<Integer>> adjacencyListGraph){
      Queue<Integer> q  = new ArrayDeque();
      q.offer(0);
      boolean[] seen = new boolean[adjacencyListGraph.size()];
      List<Integer> bfsListGraph = new ArrayList();
      while(!q.isEmpty()){
        int currentVertex = q.poll();
        bfsListGraph.add(currentVertex);
        seen[currentVertex] = true;
        List<Integer> connections = adjacencyListGraph.get(currentVertex);
        for(Integer connection: connections){          
          if(!seen[connection]) q.offer(connection);
        }
      }
      return bfsListGraph;
  }
   static List<Integer> bfsAdjacencyMatrixGraph(int[][] inputMatrix){
      Queue<Integer> q  = new ArrayDeque();
      q.offer(0);      
      boolean[] seen = new boolean[inputMatrix.length];
      List<Integer> bfsListGraph = new ArrayList();
      while(!q.isEmpty()){
        int currentVertex = q.poll();        
        bfsListGraph.add(currentVertex);
        seen[currentVertex] = true;
        int[] connections = inputMatrix[currentVertex];
        for(int i=0 ; i<connections.length; i++){
            if(connections[i]!=0 && !seen[i]) q.offer(i);            
        }
      }
      return bfsListGraph;
  }
static List<Integer> dfsAdjacencyListGraph(List<List<Integer>> adjacencyListGraph){
      List<Integer> dfsListGraph = new ArrayList();
      boolean[] seen = new boolean[adjacencyListGraph.size()];
      dfsAdjacencyListGraphHelper(adjacencyListGraph, 0, seen, dfsListGraph);
      return dfsListGraph;
}
static void dfsAdjacencyListGraphHelper(List<List<Integer>> adjacencyListGraph, int currentVertex, 
                                         boolean[] seen, List<Integer> dfsListGraph){
    dfsListGraph.add(currentVertex);
    seen[currentVertex] = true;
    List<Integer> connections = adjacencyListGraph.get(currentVertex);
    for(Integer connectionVertex : connections){        
        if(!seen[connectionVertex]) dfsAdjacencyListGraphHelper(adjacencyListGraph, connectionVertex, seen, dfsListGraph);
    }
}
static List<Integer> dfsAdjacencyMatrixGraph(int[][] inputMatrix){
    List<Integer> dfsListGraph = new ArrayList();
    boolean[] seen = new boolean[inputMatrix.length];
    dfsAdjacencyMatrixGraphHelper(inputMatrix,0, seen, dfsListGraph);
    return dfsListGraph;
}
static void dfsAdjacencyMatrixGraphHelper(int[][] inputMatrix, int currentVertex, 
                                       boolean[] seen, List<Integer> dfsListGraph){
    dfsListGraph.add(currentVertex);
    seen[currentVertex] = true;
    int[] connections = inputMatrix[currentVertex];
    for(int i=0; i< connections.length; i++){
        int connectionVertex = connections[i];
        if(connectionVertex==1 && !seen[i]) 
          dfsAdjacencyMatrixGraphHelper(inputMatrix, i,seen, dfsListGraph);
    }
}
//employee id = array index, company head id = x, each employee has one manager, i/p-manager array, index employee id
//informTime array = index is employee id, index value is time to inform employee to its direct subordinates
// subordinate in respect to graph means directed edge pointing to subordinates
static int dfsMinTimeToInform(int[] manager, int headId , int[] informTime){// headId is the starting point
    Map<Integer,List<Integer>> adjacencyMapGraph = new HashMap<>();// map of mananger/key and subordinates/value
        for(int e = 0; e < manager.length; e++){
             adjacencyMapGraph.computeIfAbsent(manager[e], emp->new ArrayList<>()).add(e);;
        //   adjacencyMapGraph.putIfAbsent(manager[e],new ArrayList<>());
        //   adjacencyMapGraph.get(manager[e]).add(e);
        }
    return dfsMinTimeToInformHelper(adjacencyMapGraph, headId, informTime);
}
static int dfsMinTimeToInformHelper(Map<Integer,List<Integer>>  adjacencyMapGraph, int currentId ,int[] informTime){
    if(!adjacencyMapGraph.containsKey(currentId))  return 0;
    List<Integer> subordinates = adjacencyMapGraph.get(currentId);     
    int max =0;
    for(Integer subordinate : subordinates){        
        max = Math.max(max,dfsMinTimeToInformHelper(adjacencyMapGraph, subordinate,informTime));
    }
    return max + informTime[currentId];
}
//courses can finish or not, prerequisite [1,0] means (0---->1) course 0 needs to taken before 1, prerequistes 
//of courses can be disconnected to means disconnected graph, so for that via brute force using bfs we need
//to encounter each vertex as starting point and check is we encounter starting point again or not
//mapGraph doesn't contain key means vertex which has no subordinates means directed edge from it
static boolean bfsCanFinishCourses(int[][] prerequisites, int noOfCourses){//just dectecting cycle from each vertex
            Map<Integer, List<Integer>> adjacencyMapGraph = new HashMap();
            boolean isCoursesFinished = false;          
            for(int[] prerequisite : prerequisites){
                adjacencyMapGraph.computeIfAbsent(prerequisite[1], course->new ArrayList<>()).add(prerequisite[0]);                
             // adjacencyMapGraph.putIfAbsent(prerequisite[1], new ArrayList());
             // adjacencyMapGraph.get(prerequisite[1]).add(prerequisite[0]);
            }
            for(Integer vertex : adjacencyMapGraph.keySet()){                 
                boolean[] seen = new boolean[noOfCourses];
                Queue<Integer> q = new ArrayDeque();
                for(Integer v : adjacencyMapGraph.get(vertex)){
                  q.offer(v);
                }                
                while(!q.isEmpty()){
                    int currentVertex = q.poll();
                    seen[currentVertex] = true;
                    if(currentVertex == vertex) return false;                    
                    if(adjacencyMapGraph.containsKey(currentVertex)){
                        List<Integer> subordinates = new ArrayList();
                        subordinates = adjacencyMapGraph.get(currentVertex);
                        for(Integer subordinate: subordinates){                            
                            if(!seen[subordinate]) q.offer(subordinate);          
                        } 
                    }
                }
            }
        return true;   
}
static boolean canFinishCoursesIndegree(int[][] prerequisites, int noOfCourses){
    int noOfIndgreeCoursesProcessed = 0;
    List<Integer> inDegreeZeroList = new ArrayList();
    Map<Integer, List<Integer>> adjacencyMapGraph = new HashMap();
    int[] inDegree = new int[noOfCourses];
    for(int[] prerequisite : prerequisites){
        adjacencyMapGraph.computeIfAbsent(prerequisite[1], course-> new ArrayList<>()).add(prerequisite[0]);        
    //    adjacencyMapGraph.putIfAbsent(prerequisite[1], new ArrayList());
    //    adjacencyMapGraph.get(prerequisite[1]).add(prerequisite[0]);
        inDegree[prerequisite[0]]++;
    }
    for(int i=0; i<inDegree.length; i++){
        if(inDegree[i]==0) inDegreeZeroList.add(i);
    }
    while(!inDegreeZeroList.isEmpty()){
      int currentVertex =inDegreeZeroList.remove(inDegreeZeroList.size()-1);
      noOfIndgreeCoursesProcessed++;
      if(!adjacencyMapGraph.containsKey(currentVertex))continue;
      List<Integer> subordinates = new ArrayList();
      subordinates = adjacencyMapGraph.get(currentVertex);
      for(Integer subordinate : subordinates){
          inDegree[subordinate]--;
          if(inDegree[subordinate]==0) inDegreeZeroList.add(subordinate);
      }
      adjacencyMapGraph.remove(currentVertex);
    }
    return noOfCourses == noOfIndgreeCoursesProcessed;
}

static int minTimeToReachSignalFromSource(int[][] networkSignalTime, int noOfVertices,
                                          int sourceVertex){//{i,j,k}-i-source,j-destination,k-signal time
  // positive weight, djisktra
  int minTimeToReachSignal = 0;
  int[] signalTimeArray = new int[noOfVertices];
  Arrays.fill(signalTimeArray, Integer.MAX_VALUE);   
  //index denotes vertex & value signal time  
  signalTimeArray[sourceVertex-1]=0;//networkSignalTime input array vertex represents from 1..n not from 0
  //minHeap is the greed, minHeap value vertex comparing signaltimearray value- distance
  Queue<Integer> minHeap = new PriorityQueue<>((a,b)-> signalTimeArray[a]-signalTimeArray[b]);
  minHeap.offer(sourceVertex-1);  
  List<int[]>[] adjacencyListGraph = new ArrayList[noOfVertices];//Position- row-destination,col-distance
  for(int i=0; i< noOfVertices; i++) adjacencyListGraph[i] = new ArrayList<int[]>();
//Map<Integer, List<int[]>> adjacencyMapGraph = new HashMap();   
  for(int[] sourceToDestination: networkSignalTime){
    adjacencyListGraph[sourceToDestination[0]-1].add(new int[]{sourceToDestination[1]-1,sourceToDestination[2]});
//  adjacencyMapGraph.putIfAbsent(sourceToDestination[0]-1, new ArrayList());
//  adjacencyMapGraph.get(sourceToDestination[0]-1).add(new int[]{sourceToDestination[1]-1,sourceToDestination[2]});
  }
  while(!minHeap.isEmpty()){
      int currentVertex = minHeap.poll();
  //  if(!adjacencyMapGraph.containsKey(currentVertex)) continue;  
      List<int[]> neighbouringVertices = new ArrayList();
  //  neighbouringVertices = adjacencyMapGraph.get(currentVertex);
      neighbouringVertices = adjacencyListGraph[currentVertex];
      for(int[] neighbouringVertexObject: neighbouringVertices){          
          int neighbouringVertex = neighbouringVertexObject[0];
          int neighbouringVertexWeight = neighbouringVertexObject[1];
          if(signalTimeArray[currentVertex] + neighbouringVertexWeight < signalTimeArray[neighbouringVertex]){
                signalTimeArray[neighbouringVertex]=signalTimeArray[currentVertex] + neighbouringVertexWeight;                
                minHeap.offer(neighbouringVertex);
          }
      }
  }  
  minTimeToReachSignal = Arrays.stream(signalTimeArray).max().getAsInt();  
  return  minTimeToReachSignal== Integer.MAX_VALUE? -1: minTimeToReachSignal;
}
static int dpMinTimeToReachSignalFromSource(int[][] networkSignalTime, int noOfVertices,
                                          int sourceVertex){//{i,j,k}-i-source,j-destination,k-signal time
  // negative  weight, bellman-ford, don't work in negative cycle,dp- what to store or memoize
  //dp-signalTimeArray- each vertex value is min cost from travelling to source to destination(is vertex)
  int minTimeToReachSignal;
  int[] signalTimeArray = new int[noOfVertices];
  Arrays.fill(signalTimeArray,Integer.MAX_VALUE);
  signalTimeArray[sourceVertex-1] = 0;
  for(int i=0; i< noOfVertices-1;i++){
      int noOfUpdation =0;
      for(int[] sourceToDestination : networkSignalTime){
          int source = sourceToDestination[0];
          int destination = sourceToDestination[1];
          int weight = sourceToDestination[2];
          if(signalTimeArray[source-1] == Integer.MAX_VALUE) continue;
          if(signalTimeArray[source-1] + weight < signalTimeArray[destination-1]){
                signalTimeArray[destination-1] = signalTimeArray[source-1] + weight;
                noOfUpdation++;
          }
      }
      if(noOfUpdation==0) break; 
  }
  minTimeToReachSignal = Arrays.stream(signalTimeArray).max().getAsInt();  
  return  minTimeToReachSignal== Integer.MAX_VALUE? -1: minTimeToReachSignal;
} 

public static void main(String args[]) {
		int[] A = {3,9,20,4,5,15,0};
//  int[] numbers = IntStream.range(0, N).map(i -> scan.nextInt()).toArray();
    List<Integer> input0 = Arrays.asList(1,3);
     List<Integer> input1 = Arrays.asList(0);
      List<Integer> input2 = Arrays.asList(3,8);
       List<Integer> input3 = Arrays.asList(0,4,5,2);
        List<Integer> input4 = Arrays.asList(3,6);
         List<Integer> input5 = Arrays.asList(3);
          List<Integer> input6 = Arrays.asList(4,7);
           List<Integer> input7 = Arrays.asList(6);
            List<Integer> input8 = Arrays.asList(2);
    List<List<Integer>> inputList = new ArrayList();
    inputList.add(input0);
    inputList.add(input1);  
    inputList.add(input2);    
    inputList.add(input3);  
    inputList.add(input4);  
    inputList.add(input5);  
    inputList.add(input6);
    inputList.add(input7);  
    inputList.add(input8);
    int[][] inputMatrix =  { {0, 1, 0, 1, 0, 0, 0, 0, 0},
                            {1, 0, 0, 0, 0, 0, 0, 0, 0},
                            {0, 0, 0, 1, 0, 0, 0, 0, 1},
                            {1, 0, 1, 0, 1, 1, 0, 0, 0},
                            {0, 0, 0, 1, 0, 0, 1, 0, 0},
                            {0, 0, 0, 1, 0, 0, 0, 0, 0},
                            {0, 0, 0, 0, 1, 0, 0, 1, 0},
                            {0, 0, 0, 0, 0, 0, 1, 0, 0},
                            {0, 0, 1, 0, 0, 0, 0, 0, 0},};
//  List<Integer> outputList = bfsAdjacencyListGraph(inputList);
//  List<Integer> outputList1 = bfsAdjacencyMatrixGraph(inputMatrix);
    List<Integer> outputList = dfsAdjacencyListGraph(inputList);
    List<Integer> outputList1 = dfsAdjacencyMatrixGraph(inputMatrix);
    System.out.println(outputList);
    System.out.println(outputList1);
    int[][] networkSignalTime = {{1,2,9},{1,4,2},{2,5,1},{4,2,4},{4,5,6},{3,2,3},{5,3,7},{3,1,5},};
    int sourceVertex = 1;
    System.out.println(minTimeToReachSignalFromSource(networkSignalTime,5, sourceVertex));
    int[][] prerequisites = {{1,0}, {2,1},{2,5}, {0,3}, {4,3}, {3,5}, {4,5},};
    System.out.println(canFinishCoursesIndegree(prerequisites, 6));
    int[] manager = {2,2,4,6,-1,4,4,5};
    int[] informTime = {0,0,4,0,7,3,6,0};
    int headId = 4;
    System.out.println("" + dfsMinTimeToInform(manager,4,informTime));
    int[][] directions = {{-1,0},{0,1},{1,0},{0,-1},};// up, right, down, left
    int[][] matrix = {{1, 2, 3, 4, 5 },
                      {6, 7, 8, 9, 10},
                      {11,12,13,14,15},
                      {16,17,18,19,20}};
    int[][] islandMatrix = {{1,1,0,0,0},
                            {1,1,0,0,0},
                            {0,0,1,0,0},
                            {0,0,0,1,1}};
    int[][] orangeMatrix = {{2,0,0,0,0},
                            {1,1,0,0,2},
                            {0,1,1,1,1},
                            {0,1,0,0,1}};
    int[][] wallGateMatrix1={{100,-1,0,100},
                            {100,100,100,-1},
                            {100,-1,100,-1},
                            {0,-1,100,100}};
    int[][] wallGateMatrix2={{100,-1,0,100},
                            {100,100,100,-1},
                            {100,-1,100,-1},
                            {0,-1,100,100}};    
    dfsMinDistanceWallGate(wallGateMatrix1);
    bfsMinDistanceWallGate(wallGateMatrix2);
    for(int i =0; i< wallGateMatrix1.length; i++){
      System.out.println(" "+ i + "th row no ");
      for(int j=0; j< wallGateMatrix1[0].length; j++){
        System.out.print( " "+ wallGateMatrix1[i][j]);
      }      
    }
    for(int i =0; i< wallGateMatrix2.length; i++){
      System.out.println(" "+ i + "th row no ");
      for(int j=0; j< wallGateMatrix2[0].length; j++){
        System.out.print( " "+ wallGateMatrix2[i][j]);
      }      
    }
    System.out.println(" island count " + dfsIslandCount(islandMatrix));
    System.out.println(" minutes to rot oranges " + bfsMinutesToRotOrange(orangeMatrix));
    TreeNode root = buildTreeLevelOrder(A);
    List<Integer> dfsList = dfsMatrix(matrix);
    System.out.println(dfsList);
    List<Integer> bfsList = bfsMatrix(matrix);
    System.out.println(bfsList);
    System.out.println(minNumberFromSequenceID("IDID"));
    
    
    /*
		Scanner in = new Scanner(System.in);
      	int num= in.nextInt(); 
      	int result=0;
      	for(int i=2;i<=num;i++){}      	
      	System.out.println(result);
     */   
}
    static int minNumberFromSequenceID(String str){
      Deque<Integer> stack = new ArrayDeque();
      String res ="";
      for(int i=0; i<=str.length(); i++){
        stack.push(i+1);
        if(i==str.length() || str.charAt(i)=='I'){
          while(!stack.isEmpty()) res += String.valueOf(stack.pop());
        }
      }
      return Integer.parseInt(res);
    }
    static Set<String> getPermutations(String inputString) {

    // base case
    if (inputString.length() <= 1) {
        return new HashSet<>(Collections.singletonList(inputString));
    }

    String allCharsExceptLast = inputString.substring(0, inputString.length() - 1);
    char lastChar = inputString.charAt(inputString.length() - 1);

    // recursive call: get all possible permutations for all chars except last
    Set<String> permutationsOfAllCharsExceptLast = getPermutations(allCharsExceptLast);

    // put the last char in all possible positions for each of the above permutations
    Set<String> permutations = new HashSet<>();
    for (String permutationOfAllCharsExceptLast : permutationsOfAllCharsExceptLast) {
        for (int position = 0; position <= allCharsExceptLast.length(); position++) {
            String permutation = permutationOfAllCharsExceptLast.substring(0, position) + lastChar
                + permutationOfAllCharsExceptLast.substring(position);
            permutations.add(permutation);
        }
    }

    return permutations;
}
    
}