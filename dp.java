 
queue
int intArray[];int itemCount;
queue(int s) intArray[] = new intArray[s]; itemCount = 0;
int peek() return intArray[itemCount - 1];
bool isEmpty() return itemCount == 0;
bool isFull() return itemCount == s;
int size() return itemCount;
void insert(int data){int i = 0;
   if(!isFull()){
      if(itemCount == 0) intArray[itemCount++] = data;        
      else{
         for(i = itemCount - 1; i >= 0; i-- )
            if(data > intArray[i]) intArray[i+1] = intArray[i];
            else break;
         intArray[i+1] = data; itemCount++;
      }
   }
}
int removeData() return intArray[--itemCount]; 
stack
 public class MyStack {
   private int maxSize; private long[] stackArray; private int top;
   public MyStack(int s) {maxSize = s;stackArray = new long[maxSize]; top = -1;}
   public void push(long j) stackArray[++top] = j;
   public long pop() return stackArray[top--];
   public long peek() return stackArray[top];
   public boolean isEmpty()  return (top == -1);
   public boolean isFull() return (top == maxSize - 1);
}
 quick sort
 void sort(int arr[], int low, int high){
        if (low < high) {
            int pi = partition(arr, low, high);
            sort(arr, low, pi-1); sort(arr, pi+1, high);
        }
    }
int partition(int arr[], int low, int high){
        int pivot = arr[high]; int i = (low-1); 
        for (int j=low; j<=high-1; j++) {
            if (arr[j] <= pivot){
                i++; int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
            }
        }
        int temp = arr[i+1]; arr[i+1] = arr[high]; arr[high] = temp;
        return i+1;
    }
 //binary search
 int binarySearch(int arr[], int x) {
        int l = 0, r = arr.length - 1;
        while (l <= r) {
            int m = l + (r-l)/2;
            if (arr[m] == x) return m;
            if (arr[m] < x) l = m + 1;
            else r = m - 1;
        }
        return -1;
    }
fibbinacci // number of binary strings without consecutive 1’s fib(n+2) 
// tiling  2*n grid ways to tile 2*1 grid  // (fib(n+2))^2 ways to construct
building  each section 2 plots on either side such that space between any two
int fib(int n){
  int f[n+1]; int i; f[0] = 0;   f[1] = 1; 
  for (i = 2; i <= n; i++) f[i] = f[i-1] + f[i-2];
  return f[n];
}
static int fib(int n){
        int a = 0, b = 1, c;
        if (n == 0)return a;
        for (int i = 2; i <= n; i++){
            c = a + b;a = b;b = c;
        }
        return b;
}
largest contiguous subarray sum // max sum rectangle in 2d matrix
int maxSubArraySum(int a[], int size){
    int max_so_far = INT_MIN, max_ending_here = 0,
       start =0, end = 0, s=0;
    for (int i=0; i< size; i++ ){
        max_ending_here += a[i];
        if (max_so_far < max_ending_here){
            max_so_far = max_ending_here;start = s;end = i;
        }
        if (max_ending_here < 0){
            max_ending_here = 0; s = i+1;
        }
    }
    cout << max_so_far<< start << end
}
max sum rectangle in 2d matrix
    public static void findMaxSubMatrix(int[][] a) {
        int cols = a[0].length; int rows = a.length;
        int[] currentResult; int maxSum = Integer.MIN_VALUE;
        int left = 0; int top = 0;int right = 0; int bottom = 0;
        for (int leftCol = 0; leftCol < cols; leftCol++) {
            int[] tmp = new int[rows];
            for (int rightCol = leftCol; rightCol < cols; rightCol++) {
                for (int i = 0; i < rows; i++)  tmp[i] += a[i][rightCol];
                currentResult = kadane(tmp);
                if (currentResult[0] > maxSum) {
                    maxSum = currentResult[0]; left = leftCol;
                    top = currentResult[1]; right = rightCol;
                    bottom = currentResult[2];
                }
            }
        }
        System.out.println( maxSum +left + top + right + bottom );
    }
 public static int[] kadane(int[] a) {
        //result[0] == maxSum, result[1] == start, result[2] == end;
        int[] result = new int[]{Integer.MIN_VALUE, 0, -1};
        int currentSum = 0; int localStart = 0;
        for (int i = 0; i < a.length; i++) { currentSum += a[i];
            if (currentSum < 0) {
                currentSum = 0; localStart = i + 1;
            } else if (currentSum > result[0]) {
                result[0] = currentSum;
                result[1] = localStart; result[2] = i;
            }
        }
        //all numbers in a are negative
        if (result[2] == -1) {result[0] = 0;
            for (int i = 0; i < a.length; i++) {
                if (a[i] > result[0]) { result[0] = a[i];
                    result[1] = i; result[2] = i;
                }
            }
        }
        return result;
    }
longest increasing subsequence // max sum // bitonic // max length chain of pairs // box stacking // building bridges no two cross
 static int lis(int arr[],int n){
          int lis[] = new int[n]; int i,j,max = 0;
           for ( i = 0; i < n; i++ ) lis[i] = 1;
              // max sum lis[i] = arr[i];
           for ( i = 1; i < n; i++ )
              for ( j = 0; j < i; j++ ) 
                         if ( arr[i] > arr[j] && lis[i] < lis[j] + 1 // lis[i] < lis[j] + arr[i] )
                    lis[i] = lis[j] + 1; // lis[i] = lis[j] + arr[i];
           for ( i = 0; i < n; i++ )
              if ( max < lis[i] ) max = lis[i];
            return max;
            // longest bitonic subsequence
             int[] lds = new int [n];
        for (i = 0; i < n; i++) lds[i] = 1;
        /* Compute LDS values from right to left */
        for (i = n-2; i >= 0; i--)
            for (j = n-1; j > i; j--)
                if (arr[i] > arr[j] && lds[i] < lds[j] + 1)
                    lds[i] = lds[j] + 1;
        /* Return the maximum value of lis[i] + lds[i] - 1*/
        int max = lis[0] + lds[0] - 1;
        for (i = 1; i < n; i++)
            if (lis[i] + lds[i] - 1 > max) max = lis[i] + lds[i] - 1;
    }
longest increasing subsequence nlogn
static int LongestIncreasingSubsequenceLength(int A[], int size){
        int[] tailTable = new int[size]; int len; 
        tailTable[0] = A[0];len = 1;
        for (int i = 1; i < size; i++){
            if (A[i] < tailTable[0]) tailTable[0] = A[i];
            else if (A[i] > tailTable[len-1]) tailTable[len++] = A[i];
            else
                tailTable[CeilIndex(tailTable, -1, len-1, A[i])] = A[i];
        }
        return len;
    }
    static int CeilIndex(int A[], int l, int r, int key) {
        while (r - l > 1){
            int m = l + (r - l)/2;
            if (A[m]>=key) r = m;
            else l = m;
        }
        return r;
    }
max length chain of pairs
struct pair {int a;int b;};
// This function assumes that arr[] is sorted in first value increasing order
int maxChainLength( struct pair arr[], int n){
   int i, j, max = 0;
   for ( i = 0; i < n; i++ ) mcl[i] = 1;
   for ( i = 1; i < n; i++ )
      for ( j = 0; j < i; j++ )
         if ( arr[i].a > arr[j].b && mcl[i] < mcl[j] + 1)
            mcl[i] = mcl[j] + 1;
   for ( i = 0; i < n; i++ ) if ( max < mcl[i] ) max = mcl[i];
   return max;
}
// box stacking
struct Box{ int h, w, d; };
int min (int x, int y) { return (x < y)? x : y; }
int max (int x, int y) { return (x > y)? x : y; }
int compare (const void *a, const void * b){
    return ( (*(Box *)b).d * (*(Box *)b).w ) -
           ( (*(Box *)a).d * (*(Box *)a).w );
}
int maxStackHeight( Box arr[], int n ){
   Box rot[3*n]; int index = 0;
   for (int i = 0; i < n; i++){
      rot[index] = arr[i]; index++;
      rot[index].h = arr[i].w;
      rot[index].d = max(arr[i].h, arr[i].d);
      rot[index].w = min(arr[i].h, arr[i].d);
      index++; rot[index].h = arr[i].d;
      rot[index].d = max(arr[i].h, arr[i].w);
      rot[index].w = min(arr[i].h, arr[i].w);
      index++;
   }
   n = 3*n;
   qsort (rot, n, sizeof(rot[0]), compare);
   int msh[n];
   for (int i = 0; i < n; i++ ) msh[i] = rot[i].h;
   for (int i = 1; i < n; i++ )
      for (int j = 0; j < i; j++ )
         if ( rot[i].w < rot[j].w &&rot[i].d < rot[j].d &&
              msh[i] < msh[j] + rot[i].h)
            	  msh[i] = msh[j] + rot[i].h;
   int max = -1;
   for ( int i = 0; i < n; i++ ) if ( max < msh[i] ) max = msh[i];
   return max;
}
longest common subsequence  // shortest common supersequence m+n- lcs
void lcs( char *X, char *Y, int m, int n ){
   int L[m+1][n+1];
   for (int i=0; i<=m; i++){
     for (int j=0; j<=n; j++){
       if (i == 0 || j == 0) L[i][j] = 0;
       else if (X[i-1] == Y[j-1]) L[i][j] = L[i-1][j-1] + 1;
       else L[i][j] = max(L[i-1][j], L[i][j-1]);
     }
   }
   int index = L[m][n]; char lcs[index+1];
   lcs[index] = '\0'; int i = m, j = n;
   while (i > 0 && j > 0){
      if (X[i-1] == Y[j-1]){
          lcs[index-1] = X[i-1];   i--; j--; index--; 
      }
      else if (L[i-1][j] > L[i][j-1]) i--;
      else j--;
   }
   cout << "LCS of " << X << " and " << Y << " is " << lcs;
}
edit distance insert remove replace
static int editDistDP(String str1, String str2, int m, int n){
        int dp[][] = new int[m+1][n+1];
        for (int i=0; i<=m; i++){
            for (int j=0; j<=n; j++){
                if (i==0) dp[i][j] = j;  // Min. operations = j
                else if (j==0) dp[i][j] = i; // Min. operations = i
                else if (str1.charAt(i-1) == str2.charAt(j-1))
                    dp[i][j] = dp[i-1][j-1];
                else
                    dp[i][j] = 1 + min(dp[i][j-1],  // Insert
                                       dp[i-1][j],  // Remove
                                       dp[i-1][j-1]); // Replace
            }
        }
  
        return dp[m][n];
}
 static int min(int x,int y,int z){
        if (x < y && x <z) return x;
        if (y < x && y < z) return y;
        else return z;
    }
min cost path from 0,0 
 private static int minCost(int cost[][], int m, int n){
        int i, j; int tc[][]=new int[m+1][n+1];
        tc[0][0] = cost[0][0];
        for (i = 1; i <= m; i++) tc[i][0] = tc[i-1][0] + cost[i][0];
        for (j = 1; j <= n; j++) tc[0][j] = tc[0][j-1] + cost[0][j];
        for (i = 1; i <= m; i++)
            for (j = 1; j <= n; j++)
                tc[i][j] = min(tc[i-1][j-1], 
                               tc[i-1][j],
                               tc[i][j-1]) + cost[i][j];
        return tc[m][n];
    }
countways coin change order doesn't matter infinite supply of coin // min coin change
static long countWays(int S[], int m, int n){
        long[] table = new long[n+1]; Arrays.fill(table, 0);   
        table[0] = 1;
        for (int i=0; i<m; i++)
            for (int j=S[i]; j<=n; j++) table[j] += table[j-S[i]];
        return table[n];
        // min coin change
        int table[V+1];
   		 table[0] = 0;
    for (int i=1; i<=V; i++) table[i] = INT_MAX;
    for (int i=1; i<=V; i++){
        for (int j=0; j<m; j++)
          if (coins[j] <= i){
              int sub_res = table[i-coins[j]];
              if (sub_res != INT_MAX && sub_res + 1 < table[i])
                  table[i] = sub_res + 1;
          }
    }
    return table[V];
    }
count ways to reach n stair using atmost m stair order matter
int countWays(int s, int m){
    return countWaysUtil(s+1, m);
}
int countWaysUtil(int n, int m){
    int res[n]; res[0] = 1; res[1] = 1;
    for (int i=2; i<n; i++){
       res[i] = 0;
       for (int j=1; j<=m && j<=i; j++) res[i] += res[i-j];
    }
    return res[n-1];
}
matrix chain multiplication 
static int MatrixChainOrder(int p[], int n){
        int m[][] = new int[n][n]; int i, j, k, L, q;
        for (i = 1; i < n; i++) m[i][i] = 0;
        for (L=2; L<n; L++){
            for (i=1; i<n-L+1; i++){
                j = i+L-1;if(j == n) continue;
                m[i][j] = Integer.MAX_VALUE;
                for (k=i; k<=j-1; k++){
                    q = m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j];
                    if (q < m[i][j]) m[i][j] = q;
                }
            }
        }
        return m[1][n-1];
}
minimum cost polygon triangulation permiter like matrix chain multiplication
struct Point{
    int x, y;
};
double min(double x, double y){
    return (x <= y)? x : y;
}
double dist(Point p1, Point p2){
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) +
                (p1.y - p2.y)*(p1.y - p2.y));
}
}
double cost(Point points[], int i, int j, int k){
    Point p1 = points[i], p2 = points[j], p3 = points[k];
    return dist(p1, p2) + dist(p2, p3) + dist(p3, p1);
}
double mTCDP(Point points[], int n){
   if (n < 3) return 0;
   double table[n][n];
   for (int gap = 0; gap < n; gap++){
      for (int i = 0, j = gap; j < n; i++, j++){
          if (j < i+2) table[i][j] = 0.0;
          else{
              table[i][j] = MAX;
              for (int k = i+1; k < j; k++){
                double val = table[i][k] + table[k][j] + cost(points,i,j,k);
                if (table[i][j] > val) table[i][j] = val;
              }
          }
      }
   }
   return  table[0][n-1];
}
binomial coefficient  ways  k  chosen from n 
static int binomialCoeff(int n, int k){
    int C[][] = new int[n+1][k+1]; int i, j;
    for (i = 0; i <= n; i++){
        for (j = 0; j <= min(i, k); j++){
            if (j == 0 || j == i) C[i][j] = 1;
            else C[i][j] = C[i-1][j-1] + C[i-1][j];
          }
     }
    return C[n][k];
}
knapsack returns max capacity W
static int knapSack(int W, int wt[], int val[], int n){
     int i, w; int K[][] = new int[n+1][W+1];
   for (i = 0; i <= n; i++){
      for (w = 0; w <= W; w++){
         if (i==0 || w==0) K[i][w] = 0;
         else if (wt[i-1] <= w) K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w]);
         else K[i][w] = K[i-1][w];
      }
    }
      return K[n][W];
 }
egg dropping puzzle
 static int eggDrop(int n, int k){
        int eggFloor[][] = new int[n+1][k+1];
        int res; int i, j, x;
      for (i = 1; i <= n; i++) eggFloor[i][1] = 1; eggFloor[i][0] = 0;
        for (j = 1; j <= k; j++)eggFloor[1][j] = j;
        for (i = 2; i <= n; i++){
            for (j = 2; j <= k; j++){
                eggFloor[i][j] = Integer.MAX_VALUE;
                for (x = 1; x <= j; x++){
                     res = 1 + max(eggFloor[i-1][x-1], eggFloor[i][j-x]);
                     if (res < eggFloor[i][j]) eggFloor[i][j] = res;
                }
            }
        }
        return eggFloor[n][k];
    }
longest palindromic subsequence
 static int lps(String seq){
       int n = seq.length(); int i, j, cl;
       int L[][] = new int[n][n]; 
       for (i = 0; i < n; i++) L[i][i] = 1;
        for (cl=2; cl<=n; cl++){
            for (i=0; i<n-cl+1; i++){
                j = i+cl-1;
                if (seq.charAt(i) == seq.charAt(j) && cl == 2) L[i][j] = 2;
                else if (seq.charAt(i) == seq.charAt(j)) L[i][j] = L[i+1][j-1] + 2;
                else  L[i][j] = max(L[i][j-1], L[i+1][j]);
            }
        }
        return L[0][n-1];
    }
cutting a rod
 static int cutRod(int price[],int n){
        int val[] = new int[n+1];val[0] = 0;
        for (int i = 1; i<=n; i++){
            int max_val = Integer.MIN_VALUE;
            for (int j = 0; j < i; j++)
                max_val = Math.max(max_val, 
                                   price[j] + val[i-j-1]);
            val[i] = max_val;
        }
        return val[n];
    }
palindrome partitioning minimum cut like matrix chain multiplication
int minPalPartion(char *str){
    int n = strlen(str);int C[n];
    bool P[n][n];int i, j, k, L; 
    for (i=0; i<n; i++)P[i][i] = true;
    for (L=2; L<=n; L++){
        for (i=0; i<n-L+1; i++){
            j = i+L-1; 
            if (L == 2) P[i][j] = (str[i] == str[j]);
            else P[i][j] = (str[i] == str[j]) && P[i+1][j-1];
        }
	}
	for (i=0; i<n; i++){
		if (P[0][i] == true) C[i] = 0;
		else{
			C[i] = INT_MAX;
			for(j=0;j<i;j++)
				if(P[j+1][i] == true && 1+C[j]<C[i]) C[i]=1+C[j];
		}
	}
    return C[n-1];
}
partition problem subset divided into half
    static boolean findPartition (int arr[], int n){
        int sum = 0;int i, j;
        for (i = 0; i < n; i++)sum += arr[i];
        if (sum%2 != 0)return false;
        boolean part[][]=new boolean[sum/2+1][n+1];
        for (i = 0; i <= n; i++)part[0][i] = true;
        for (i = 1; i <= sum/2; i++) part[i][0] = false;
        for (i = 1; i <= sum/2; i++){
            for (j = 1; j <= n; j++){
                part[i][j] = part[i][j-1];
                if (i >= arr[j-1])
                    part[i][j] = part[i][j] ||
                                 part[i - arr[j-1]][j-1];
            }
        }
        for (i = 0; i <= sum/2; i++){
            for (j = 0; j <= n; j++)printf ("%4d", part[i][j]);
        }
        return part[sum/2][n];
    }
word wrap problem put line breaks so that spaces will be minimum
void solveWordWrap (int l[], int n, int M){
    int extras[n+1][n+1];  int lc[n+1][n+1];
    int c[n+1];int p[n+1];int i, j;
    for (i = 1; i <= n; i++){
        extras[i][i] = M - l[i-1];
        for (j = i+1; j <= n; j++) extras[i][j] = extras[i][j-1] - l[j-1] - 1;
    }
    for (i = 1; i <= n; i++){
        for (j = i; j <= n; j++){
            if (extras[i][j] < 0)lc[i][j] = INF;
            else if (j == n && extras[i][j] >= 0)lc[i][j] = 0;
            else lc[i][j] = extras[i][j]*extras[i][j];
        }
    }
    c[0] = 0;
    for (j = 1; j <= n; j++){
        c[j] = INF;
        for (i = 1; i <= j; i++){
            if (c[i-1] != INF && lc[i][j] != INF && (c[i-1] + lc[i][j] < c[j])){
                c[j] = c[i-1] + lc[i][j];p[j] = i;
            }
        }
    }
    printSolution(p, n);
}

int printSolution (int p[], int n){
    int k;
    if (p[n] == 1)k = 1;
    else k = printSolution (p, p[n]-1) + 1;
    return k;
}
min no of jumps to reach end
int minJumps(int arr[], int n){
    int *jumps = new int[n]; int i, j;
    if (n == 0 || arr[0] == 0)return INT_MAX;
    jumps[0] = 0;
    for (i = 1; i < n; i++){
        jumps[i] = INT_MAX;
        for (j = 0; j < i; j++){
            if (i <= j + arr[j] && jumps[j] != INT_MAX){
               jumps[i] = min(jumps[i], jumps[j] + 1);break;
            }
        }
    }
    return jumps[n-1];
}
max size sub square matrix with all 1's
void printMaxSubSquare(bool M[R][C]){
  int i,j;int S[R][C];int max_of_s, max_i, max_j; 
  for(i = 0; i < R; i++)S[i][0] = M[i][0];
  for(j = 0; j < C; j++)S[0][j] = M[0][j];
  for(i = 1; i < R; i++){
    for(j = 1; j < C; j++){
      if(M[i][j] == 1) S[i][j] = min(S[i][j-1], S[i-1][j], S[i-1][j-1]) + 1;
      else S[i][j] = 0;
    }    
  } 
  max_of_s = S[0][0]; max_i = 0; max_j = 0;
  for(i = 0; i < R; i++){
    for(j = 0; j < C; j++){
      if(max_of_s < S[i][j]){
         max_of_s = S[i][j];max_i = i; max_j = j;
      }        
    }                 
  }     
ugly numbers prime factors 2,3,5 find nth
unsigned getNthUglyNo(unsigned n){
    unsigned ugly[n]; // To store ugly numbers
    unsigned i2 = 0, i3 = 0, i5 = 0;
    unsigned next_multiple_of_2 = 2;
    unsigned next_multiple_of_3 = 3;
    unsigned next_multiple_of_5 = 5;
    unsigned next_ugly_no = 1;
    ugly[0] = 1;
    for (int i=1; i<n; i++){
       next_ugly_no = min(next_multiple_of_2,
                           min(next_multiple_of_3,
                               next_multiple_of_5));
       ugly[i] = next_ugly_no;
       if (next_ugly_no == next_multiple_of_2){
           i2 = i2+1;
           next_multiple_of_2 = ugly[i2]*2;
       }
       if (next_ugly_no == next_multiple_of_3){
           i3 = i3+1;
           next_multiple_of_3 = ugly[i3]*3;
       }
       if (next_ugly_no == next_multiple_of_5){
           i5 = i5+1;
           next_multiple_of_5 = ugly[i5]*5;
       }
    } 
    return next_ugly_no;
}
longest palindromic substring
int longestPalSubstr( char *str ){
    int n = strlen( str );  bool table[n][n];
    int maxLength = 1;
    for (int i = 0; i < n; ++i)table[i][i] = true;
    int start = 0;
    for (int i = 0; i < n-1; ++i){
        if (str[i] == str[i+1]){
            table[i][i+1] = true;
            start = i; maxLength = 2;
        }
    }
    for (int k = 3; k <= n; ++k){
        for (int i = 0; i < n-k+1 ; ++i){
            int j = i + k - 1;
            if (table[i+1][j-1] && str[i] == str[j]){
                table[i][j] = true;
                if (k > maxLength){
                    start = i; maxLength = k;
                }
            }
        }
    }
    printSubStr( str, start, start + maxLength - 1 );
    return maxLength; // return length of LPS
}
optimal binary search tree cost like matrix chain multiplication
int optimalSearchTree(int keys[], int freq[], int n){
    int cost[n][n];
    for (int i = 0; i < n; i++)cost[i][i] = freq[i];
    for (int L=2; L<=n; L++){
        for (int i=0; i<=n-L+1; i++){
            int j = i+L-1;
            cost[i][j] = INT_MAX;
            for (int r=i; r<=j; r++){
               int c = ((r > i)? cost[i][r-1]:0) + 
                       ((r < j)? cost[r+1][j]:0) + 
                       sum(freq, i, j);
               if (c < cost[i][j])cost[i][j] = c;
            }
        }
    }
    return cost[0][n-1];
}
int sum(int freq[], int i, int j){
    int s = 0;
    for (int k = i; k <=j; k++)s += freq[k];
    return s;
}
boolean parenthise to true no of ways
int countParenth(char symb[], char oper[], int n){
    int F[n][n], T[n][n];
    for (int i = 0; i < n; i++){
        F[i][i] = (symb[i] == 'F')? 1: 0;
        T[i][i] = (symb[i] == 'T')? 1: 0;
    }
    for (int gap=1; gap<n; ++gap){
        for (int i=0, j=gap; j<n; ++i, ++j){
            T[i][j] = F[i][j] = 0;
            for (int g=0; g<gap; g++){
                int k = i + g;
                int tik = T[i][k] + F[i][k];
                int tkj = T[k+1][j] + F[k+1][j];
                if (oper[k] == '&'){
                    T[i][j] += T[i][k]*T[k+1][j];
                    F[i][j] += (tik*tkj - T[i][k]*T[k+1][j]);
                }
                if (oper[k] == '|'){
                    F[i][j] += F[i][k]*F[k+1][j];
                    T[i][j] += (tik*tkj - F[i][k]*F[k+1][j]);
                }
                if (oper[k] == '^'){
                    T[i][j] += F[i][k]*T[k+1][j] + T[i][k]*F[k+1][j];
                    F[i][j] += T[i][k]*T[k+1][j] + F[i][k]*F[k+1][j];
                }
            }
        }
    }
    return T[0][n-1];
}
mobile numeric keypad problem 4*3 grid with escape * and # up left right down
no of possible no's of given length n
int getCount(char keypad[][3], int n){
    if(keypad == NULL || n <= 0) return 0;
    if(n == 1) return 10;
    int row[] = {0, 0, -1, 0, 1};
    int col[] = {0, -1, 0, 1, 0};
    int count[10][n+1];
    int i=0, j=0, k=0, move=0, ro=0, co=0, num = 0;
    int nextNum=0, totalCount = 0;
    for (i=0; i<=9; i++){
        count[i][0] = 0;count[i][1] = 1;
    }
    for (k=2; k<=n; k++){
        for (i=0; i<4; i++) {
            for (j=0; j<3; j++)  {
                if (keypad[i][j] != '*' && keypad[i][j] != '#'){
                    num = keypad[i][j] - '0';count[num][k] = 0;
                    for (move=0; move<5; move++){
                        ro = i + row[move];co = j + col[move];
                        if (ro >= 0 && ro <= 3 && co >=0 && co <= 2 &&
                           keypad[ro][co] != '*' && keypad[ro][co] != '#'){
                            nextNum = keypad[ro][co] - '0';
                            count[num][k] += count[nextNum][k-1];
                        }
                    }
                }
            }
        }
    }
    totalCount = 0;
    for (i=0; i<=9; i++)totalCount += count[i][n];
    return totalCount;
}
count of n digit no's whose sum equal to s leading 0 not allowed
 static int lookup[][] = new int[101][50001];
    static int countRec(int n, int sum){
        if (n == 0)return sum == 0 ? 1 : 0;
        if (lookup[n][sum] != -1)return lookup[n][sum];
        int ans = 0;
        for (int i=0; i<10; i++)
           if (sum-i >= 0)ans += countRec(n-1, sum-i);
        return lookup[n][sum] = ans;
    }
    static int finalCount(int n, int sum){
        for(int i = 0;i<=100;++i){
            for(int j=0;j<=50000;++j){
                lookup[i][j] = -1;
            }
        }
        int ans = 0;
        for (int i = 1; i <= 9; i++)
          if (sum-i >= 0)ans += countRec(n-1, sum-i);
        return ans;
    }
min poinst to reach dest cons down right no move i,j if overall points <=0
 static int minInitialPoints(int points[][],int R,int C){
        int dp[][] = new int[R][C]; int m = R, n = C;
        dp[m-1][n-1] = points[m-1][n-1] > 0? 1:
                       Math.abs(points[m-1][n-1]) + 1;
        for (int i = m-2; i >= 0; i--)
             dp[i][n-1] = Math.max(dp[i+1][n-1] - points[i][n-1], 1);
        for (int j = n-2; j >= 0; j--)
             dp[m-1][j] = Math.max(dp[m-1][j+1] - points[m-1][j], 1);
        for (int i=m-2; i>=0; i--){
            for (int j=n-2; j>=0; j--){
                int min_points_on_exit = Math.min(dp[i+1][j], dp[i][j+1]);
                dp[i][j] = Math.max(min_points_on_exit - points[i][j], 1);
            }
         }
         return dp[0][0];
    }
count non decreas no's with n digits 223, 4423332
 static int countNonDecreasing(int n){
        int dp[][] = new int[10][n+1];
        for (int i = 0; i < 10; i++)dp[i][1] = 1;
        for (int digit = 0; digit <= 9; digit++){
            for (int len = 2; len <= n; len++){
                for (int x = 0; x <= digit; x++)
                    dp[digit][len] += dp[x][len-1];
            }
        }
        int count = 0;
        for (int i = 0; i < 10; i++)count += dp[i][n];
        return count;
    }
longest path in char matrix move 8 direction alphabetical order
    static int x[] = {0, 1, 1, -1, 1, 0, -1, -1};
    static int y[] = {1, 0, 1, 1, -1, -1, 0, -1};
    static int R = 3;static int C = 3;
    static int dp[][] = new int[R][C];
    static boolean isvalid(int i, int j){
        if (i < 0 || j < 0 || i >= R || j >= C)
          return false;
        return true;
    }
    static boolean isadjacent(char prev, char curr){
        return ((curr - prev) == 1);
    }
    static int getLenUtil(char mat[][], int i, int j, char prev){
        if (!isvalid(i, j) || !isadjacent(prev, mat[i][j]))
             return 0;
        if (dp[i][j] != -1)return dp[i][j];
        int ans = 0; 
        for (int k=0; k<8; k++)
          ans = Math.max(ans, 1 + getLenUtil(mat, i + x[k],
                                       j + y[k], mat[i][j]));
        return dp[i][j] = ans;
    }
    static int getLen(char mat[][], char s){
        for(int i = 0;i<R;++i)
            for(int j = 0;j<C;++j)dp[i][j] = -1;
        int ans = 0;
        for (int i=0; i<R; i++){
            for (int j=0; j<C; j++){
                if (mat[i][j] == s) {
                    for (int k=0; k<8; k++)
                      ans = Math.max(ans, 1 + getLenUtil(mat,
                                        i + x[k], j + y[k], s));
                }
            }
        }
        return ans;
    }
min no of squares whose sum equal to n
	static int getMinSquares(int n){
	    int dp[] = new int[n+1];dp[0] = 0;
	    dp[1] = 1;dp[2] = 2;dp[3] = 3;
	    for (int i = 4; i <= n; i++){
	        dp[i] = i;
	        for (int x = 1; x <= i; x++) {
	            int temp = x*x;
	            if (temp > i) break;
	            else dp[i] = Math.min(dp[i], 1+dp[i-temp]);
	        }
	    }
	    int res = dp[n];
	    return res;
	}
max point two traverse cross diag and down top left - left bottom, top right - right bottom
int geMaxCollection(int arr[R][C]){
    int mem[R][C][C];
    return getMaxUtil(arr, mem, 0, 0, C-1);
}
bool isValid(int x, int y1, int y2){
    return (x >= 0 && x < R && y1 >=0 &&
            y1 < C && y2 >=0 && y2 < C);
}
int getMaxUtil(int arr[R][C], int mem[R][C][C], int x, int y1, int y2){
    if (!isValid(x, y1, y2)) return INT_MIN;
    if (x == R-1 && y1 == 0 && y2 == C-1)
       return (y1 == y2)? arr[x][y1]: arr[x][y1] + arr[x][y2];
    if (x == R-1) return INT_MIN;
    if (mem[x][y1][y2] != -1) return mem[x][y1][y2];
    int ans = INT_MIN;
    int temp = (y1 == y2)? arr[x][y1]: arr[x][y1] + arr[x][y2];
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1, y2-1));
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1, y2+1));
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1, y2));
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1-1, y2));
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1-1, y2-1));
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1-1, y2+1));
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1+1, y2));
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1+1, y2-1));
    ans = max(ans, temp + getMaxUtil(arr, mem, x+1, y1+1, y2+1));
    return (mem[x][y1][y2] = ans);
}
max profit buy sell atmost twice
static int maxProfit(int price[], int n){
        int profit[] = new int[n];
        for (int i=0; i<n; i++)profit[i] = 0;
        int max_price = price[n-1];
        for (int i=n-2;i>=0;i--){
            if (price[i] > max_price) max_price = price[i];
            profit[i] = Math.max(profit[i+1], max_price-price[i]);
        }
        int min_price = price[0];
        for (int i=1; i<n; i++){
            if (price[i] < min_price) min_price = price[i];
            profit[i] = Math.max(profit[i-1], profit[i] +
                                        (price[i]-min_price) );
        }
        int result = profit[n-1];
        return result;
    }
max no using 4 keys and n strokes
int findoptimal(int N){
    if (N <= 6) return N;
    int screen[N];int b; int n;
    for (n=1; n<=6; n++)screen[n-1] = n;
    for (n=7; n<=N; n++){
        screen[n-1] = 0;
        for (b=n-3; b>=1; b--){
            int curr = (n-b-1)*screen[b-1];
            if (curr > screen[n-1])screen[n-1] = curr;
        }
    }
    return screen[N-1];
}
min cost to reach dest by train
    static int minCost(int cost[][] , int N){
        int dist[] = new int[N];
        for (int i=0; i<N; i++) dist[i] = INF;
        dist[0] = 0;
        for (int i=0; i<N; i++)
           for (int j=i+1; j<N; j++)
              if (dist[j] > dist[i] + cost[i][j])
                 dist[j] = dist[i] + cost[i][j];
      
        return dist[N-1];
    }
weight job scheduling
class Job{
    int start, finish, profit;
    Job(int start, int finish, int profit){
        this.start = start;this.finish = finish;
        this.profit = profit;
    }
}
class JobComparator implements Comparator<Job>{
    public int compare(Job a, Job b){
        return a.finish < b.finish ? -1 : a.finish == b.finish ? 0 : 1;
    }
}
public class WeightedIntervalScheduling{
    static public int binarySearch(Job jobs[], int index){
        int lo = 0, hi = index - 1;
        while (lo <= hi){
            int mid = (lo + hi) / 2;
            if (jobs[mid].finish <= jobs[index].start){
                if (jobs[mid + 1].finish <= jobs[index].start)
                    lo = mid + 1;
                else return mid;
            }
            else hi = mid - 1;
        }
        return -1;
    }
    static public int schedule(Job jobs[]){
        Arrays.sort(jobs, new JobComparator());
        int n = jobs.length;int table[] = new int[n];
        table[0] = jobs[0].profit;
        for (int i=1; i<n; i++){
            int inclProf = jobs[i].profit;
            int l = binarySearch(jobs, i);
            if (l != -1)inclProf += table[l];
            table[i] = Math.max(inclProf, table[i-1]);
        }
        return table[n-1];
    }
longest even length substring such first half and second half sum same
int findLength(string str, int n){
    int ans = 0; // Initialize result
    for (int i = 0; i <= n-2; i++){
        int l = i, r = i + 1;int lsum = 0, rsum = 0;
        while (r < n && l >= 0)
		{   lsum += str[l] - '0';
            rsum += str[r] - '0';
            if (lsum == rsum)ans = max(ans, r-l+1);
            l--;r++;
        }
    }
    return ans;
}
 

