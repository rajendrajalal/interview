import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;
// s-string,A-array or node,hs-hashset,hm-hasmap,at-arraylst, dt-linkedlist, dq-deque, pq-priroity queue, st-stack, qe-queue
// res-output,v-value or key,n-string or array length,M-max,m-min,f-front,e-end, change all code to java8
public class Source {
   public static void main(String args[]) {
     Scanner in = new Scanner(System.in);
      int num= in.nextInt(); int result=0;
      for(int i=2;i<=num;i++){if(isPalindrome(Integer.toBinaryString(i)))result+=i;}
      System.out.println(result);
    }
private static boolean stringDataType(String str) {
return (str == null || str.length() <= 1) || str.equals(new StringBuilder(str).reverse().toString());
/*false binary palindrome*/for (int i = 0, j = str.length() - 1; i < j; ++i, --j) if (str.charAt(i) != str.charAt(j))
/*reverse word order*/for (int i = inputringArray.length - 1; i > 0; i--) emptyArray += inputringArray[i] + " ";
/*reverse each word*/for (int i = 0; i < str.length; i++) str[i] = new StringBuilder(str[i]).reverse().toString();
/*reversestr by offset k*/k = k % n;reverse(charArray, 0, n - k - 1); reverse(charArray, n - k, n - 1); reverse(charArray, 0, n - 1);
/*is string alphabetical*/if (!Character.isLetter(str.charAt(i)) || !(str.charAt(i) <= str.charAt(i + 1))) return false;
/*is anagram*/Arrays.sort(chr1Arr); Arrays.sort(chr2Arr);return new String(chr1Arr).equals(new String(chr2Arr));
/*J="aA", S="aAAbbbb" output 3*/for(char c : str1.toCharArray()) if(str2.indexOf(c) != -1) res0++;
countCharacters(String str) return str.replaceAll("\\s", "").length();
wordCount(String s) {if (s == null || s.isEmpty()) return 0;  return s.trim().split("[\\s]+").length;

}

if word_a[i] == word_b[j]: The letters match.
cell[i][j] = cell[i-1][j-1] + 1
else: The letters don’t match.
cell[i][j] = max(cell[i-1][j], cell[i][j-1])


private static boolean arrayDataType(int[] A,int k,int m,int B[],int n,int sum) { // m & n are no of elements,k diff input
/*maxsubarraySum*/for(i<A.len){sum0 += A[i];M_MIN_VALUE = Math.max(M, sum0 - min0);min0 = Math.min(min0, sum0);}return M;
/*minSubArrayLenSumEqualsK*/for(i<A.len){sum0+=A[i];while (sum0>=k){M_MAX_VALUE=Math.min(M, i+1-f);sum-=A[f++];}}return M==Integer.MAX_VALUE? 0: M;
/*mergearrayinone*/ int i=m-1;j=n-1;k=m+n-1;while(k>=0) A[k--] = (j < 0 || (i >= 0 && A[i] > B[j]))? A[i--]: B[j--];
/*maxProductValueArray*/for (i<A.len) {if (A[i] < 0) { int temp = maxA0; maxA0 = minA0; minA0 =temp;}
maxA0 = Math.max(A[i], maxA0 * A[i]);minA0 = Math.min(A[i], minA0 * A[i]);resA0 = Math.max(resA0, maxA0);}return resA0;}
/*twopairsumcountarray*/for (i<A.len){for (int j = i + 1; j < n; j++) if (A[i]+A[j] == sum) res0++;} return res0;
/*return intersection of two arrays*/set.add(A[i<A.len]);if (set.contains(B[j<B.len])) hs.add(B[j<B.len]);
for (Integer hsv : hs) result[k0++] = hsv; return res[hs.size()];
peakIndexInArray(int[] A) {int i = 0; while (A[i] < A[i+1]) i++; return i;}
findPeakElementAnyIndex(int[] A) {int i = 0, j = A.length - 1;
while (i < j) {int mid = i + (j - i) / 2;  if (A[mid] > A[mid + 1]) j = mid;else i = mid + 1;}return A[i];
int majority(int[] nums) {int count=0,res=0;for(int num : nums) {if(count==0)res=num;count+=(num==res)?1:-1;}return res;
pivotIndexSumLeftRightEqual(int[] nums){ int sum = 0, left = 0,n=nums.length;for (int x: nums) sum += x;
for (i<n) {  if (left == sum - left - nums[i]) return i;  left += nums[i];}return -1;
int partitionArrayKValueLeftRight(int[] nums, int k) {int offset = 0,temp=0, n= nums.length;
for (i<n) {  if (nums[i] < k) {temp = nums[i];  nums[i] = nums[offset];  nums[offset] = temp;  offset ++;}}return offset;
long factorial(int n) {if (n < 0) throw new IllegalArgumentException("n");return n == 0 || n == 1 ? 1 : n * factorial(n - 1);}
sumAllDivisors(int number) {int sum = 0;for(int i=1, limit=number/2; i<=limit; ++i) if(number % i==0) sum += i;return sum;
maxElementVal(int[] array, int len) {return len == 1 ? array[0] : Math.max(max(array, len - 1), array[len - 1]);}
isPerfectCube(int number) {  int a = (int) Math.pow(number, 1.0 / 3);return a * a * a == number;}
ifPowerOfTwoOrNot(int number) {return number != 0 && ((number & (number - 1)) == 0);}
isPrime(int num) {for (int div = 3; div <= Math.sqrt(num); div += 2) if (num % div == 0) return false;  return true;}
sumOfDigits(int n) {int sum; for (sum = 0; n > 0; sum += n % 10, n /= 10); }return sum;
sumDigits(int n)   return n == 0 ? 0 : n%10 +  sumDigits(n/10) ;
singleNumberOtherTwice(int[] A) {​  int result = 0;for (int Av : A) result ^= Av;return result;
canJump(int[] nums){int n=nums.len,p=n-1;for(int i=n-1; i>=0; i--) if (i + nums[i] >= p) p = i;return p == 0;}
minNoOfJump(int[] A){int res=0,p=0,q=0;for(i<A.len-1){ p=Math.max(p, i + A[i]);if(i==q){jumps++; q=p;}return res;
distributeCandiesInTwo(int[] n){hs_Integer;for (int num: n) hs.add(num);return hs.size()<n.len/2?hs.size():n.len/2;}
minCandyDistribute(int[] ratings) { int[] candies = new int[ratings.length];  Arrays.fill(candies, 1); int n=ratings.len, res=0;
for (i<n-1;) { if (ratings[i] >= ratings[i+1]) continue;candies[i+1] = candies[i] + 1;}
for (int i=n-1; i>0; i--) {  if (ratings[i] >= ratings[i-1]) continue;candies[i-1] = Math.max(candies[i] + 1, candies[i-1]);}
for (int i=0; i<candies.length; i++) result += candies[i];return res;
noOfCombinationsChange(int[] coins, int amount) {int[] comb = new int[amount + 1]; comb[0] = 1;
for (int coin : coins) {for (int i = coin; i < amount + 1; i++) comb[i] += comb[i - coin];}return comb[amount];
minCoinChange(int[] coins, int amount){int[] dp=new int[amount+1];Arrays.fill(dp, Integer.MAX_VALUE);
for(i<=amount){if(dp[i]==Integer.MAX_VALUE)continue;{for(int coin:coins){if(i<=amount-coin)dp[i+coin]=Math.min(dp[i]+1,dp[i+coin]);}}
return dp[amount]==Integer.MAX_VALUE ? -1: dp[amount];
sortArray0_1_2(int[] nums){int[] buckets = new int[3],j=0;  for(int num : nums) buckets[num]++;
for(int val = 0; val < 3; val++) {for(int i = 0; count < buckets[val]; i++) nums[j++] = val;}}
canPlaceFlowersNotAdjacent(int[] flowerbed/A, int n) {for(i<A.len)
{if(A[i] == 0 && (i == 0 || A[i - 1] == 0) &&(i == A.len - 1 || A[i + 1] == 0)) {A[i] = 1; n--;}
if (n <= 0) return true;}return false;
moveZeroesEnd(int[] n){int count=0; for(i<n.len) if(n[i] != 0) n[count++] = n[i];while (count < n) n[count++] = 0;}
removeDuplicates(int[] A) { Arrays.sort(A);int j=0,i=1;  while (i < A.len) {if (A[i] != A[j]) {  j++;  A[j] = A[i];}i++;}return j+1;
 /* Arrays.sort(nums); return nums[n-k]*/
findKthLargest(int[] nums, int k){pq_Integer;for(int val:nums){pq.offer(val);if(pq.size() > k) pq.poll();}return pq.peek();
 // [100, 4, 200, 1, 3, 2] return 4, subsequence [1, 2, 3, 4]
 longestConsecutiveSubsequenceLen(int[] nums){hs_Integer;int result = 0;for(int i : nums) hs.add(i);
 for(int num : nums) {  int p = num - 1, q = num + 1;  while(hs.remove(p)) p--;while(hs.remove(q)) q++;
 ans = Math.max(ans,q - p - 1);  if(hs.isEmpty()) return res;}return res;
 longestIncreasingSubsequenceLen(int[] nums){int i,j,result = 0,n=nums.length; int[] temp=new int[n]; Arrays.fill(temp,1);
 for (i_1 < n) {for (j < i)  if (nums[i] > nums[j] && temp[i] < temp[j] + 1) temp[i] = temp[j] + 1;}
 for (i = 0; i < n; i++) if (max < temp[i]) max = temp[i];return max;
 countSetBits(int num) {int cnt = 0;while (num != 0) {num = num & (num - 1);  cnt++;}return cnt;
 isContainDuplicate(int[] n){if(n==null||n.len==0)return false;hs_Integer;for(int i: n)if(!hs.add(i)) return true;return false;
}




private static boolean linkedlistDataType(Node A,Node B){// m & n are no of elements
/*mergetwolist*/if (A.v<B.v) { A.n = mergeTwoList(A.n, B);return A;}else {B.n = mergeTwoList(A, B.n); return B;}
Node oddEvenList(Node A) {if (A != null) {​ Node f = A, f1 = A.n, f2 = f1;
​while (f1 != null && f1.n != null) {f.n = f.n.n; f1.next = f.n.n; f = f.n; f1 = f1.n;}f.n = f2;}return A;
Node reverseList(Node A){if (A == null || A.next == null) return A;Node p = reverseList(A.n); A.n.n = A; A.n = null;return p;
Node intersection(Node A, Node B){Node p=A,q=B;while(p!=q){p =(p!=null) ? p.n : B;q= (q != null) ? q.n : A;}return p;
Node middleNode(Node A) {Node p = A, q = A;while (q != null && q.n != null) {p = p.n; q = q.n.n;}return p;
Node deleteDuplicate(Node A) {Node p=A;while (p != null && p.n != null) {if (p.n.v == p.v) p.n = p.n.n; else p = p.n;}return A;
Node removeElementVal(Node A, int v)Node dummy = new Node(0),curr = A, prev = dummy; dummy.n = A;
while (curr != null) {if (curr.val == val) prev.n = curr.n;else prev = prev.n;curr = curr.n;}return dummy.n
Node removeNthEndNode(Node A, int n) {Node p, q, dummy = new Node(0);dummy.n = A; p = A; q = dummy;
for (i<n) p = p.n;while (p != null) {p = p.n; q = q.n;}  q.n = q.n.n;   return dummy.n;
isPalindromeList(Node A) {Node mid=middleNode(A); mid=reverseList(mid);
while (A != null && mid != null) {  if (A.v != mid.v) return false;  A = A.n; mid = mid.n;}return true;
Node detectCycle(Node head) { if (head == null || head.next == null) return null; Node slow = head; Node fast = head;
while (slow != null && fast != null) {  slow = slow.next;if (fast.next == null) return null;fast = fast.next.next;if (slow == fast) break;}
slow = head; if (fast != null) {while (slow != fast) {  slow = slow.next;  fast = fast.next;}return slow;}return null;


}



private static boolean treeDataType{//Node{int value,Node left,Node right, Node parent,in(root.left),print root.v,in(root.right))}
in(Node root, List<Integer> result){in(root.left, result); result.add(root.val); in(root.right, result);}
/*depth*/1+Math.max(depth(root.left),depth(root.right));
/*issymmetric*/ return root==null || isMirror(root.left,root.right);
/*ismirror*/return isMirror(node1.left, node2.right) && isMirror(node1.right, node2.left);
diameterBT(Node root) {depth(root);return max_c;}​depth(Node root) {left/right = maxDepth(root.l);
max_c = Math.max(max, left + right);return Math.max(left, right) + 1;}
/*leveltp*/level(Node root) {res_list_Alist_Integer;Util(root, 0/d, res); return res;/*Util*/Util(root,depth,res_list_list_Integer)
if(d==res.size())res.add(new ArrayList<Integer>());res.get(d).add(root.val);Util(root.left,d+1,res);Util(root.right,d+1,res);
/*levelbp*/level(Node root) {res_list_Llist_Integer;Util(root, 0/d, res); return res;/*Uil*/Util(root,depth,res_list_list_Integer)
if(d>=res.size())res.add(new ArrayList<Integer>());Util(root.l,d+1,res);Util(root.r,d+1,res);res.get(res.size()-d -1).add(root.v);
/*levelZ*/levelZ(Node root) {res_list_Alist_Integer;Queue<Node> q= new LinkedList<>(); q.offer(root);zigzag= false;
while(!q.isEmpty()) {level_Alist_Integer;size = q.size();for (i<size) {Node node=q.poll();
if (zigzag) level.add(0, node.v);else level.add(node.v);if(node.l!=null) q.add(node.l);if(node.r!=null)q.add(node.r);}
res.add(level);zigzag = !zigzag;}return res;
/*vertical*/vertical(Node root){TreeMap<Integer,Vector<Integer>> m = new TreeMap<>();int hd =0;Util(root,hd,m);
Util{Vector<Integer> get= m.get(hd);if(get==null){get=new Vector<>();get.add(root.key);} else{ get.add(root.key);}
m.put(hd, get);Util(root.left, hd-1, m);  Util(root.right, hd+1, m);}
/*isbst*/isBST(Node root)return isBSTUtil(root, Integer.MIN_VALUE, Integer.MAX_VALUE); /*isbstutil*/
if(root.v<min || root.v > max)return false;return isValid(root.left,min,root.v-1) && isValid(root.right,root.v-1,max);
/*haspathsum*/if(root.v==sum&&root.l==null&&root.r==null)return true;return hasPathSum(root.l,sum-root.v)||hasPathSum(root.r,sum-root.v);
noOfPathSum(Node root, int sum) { return path(root, sum) + noOfPathSum(root.l, sum) + noOfPathSum(root.r, sum);}
path(Node root, int sum) {return (root.v==sum ? 1 : 0) + path(root.l,sum-root.v) + path(root.r,sum-root.v);}
maxPathSum(Node root) {Util(root);return maxSum_c;} Util(Node root) {lPath/rpath = Math.max(Util(root.l), 0);
maxSum_c = Math.max(maxSum_c, root.v + lPath + rPath);return root.v + Math.max(lPath, rPath);}
isSameTree(Node p, Node q) { /*node1&node2 true,node1ornode false*/ return p.v == q.v && isSameTree(p.l, q.l) && isSameTree(p.r, q.r);}
nodedistance(root,node p,node q){Node lca=lcaBT(root,p,q);int d1/d2=dist(lca,p/q);if(d1<0||dt2<0)return -1;return d1+d2;
Node lcaBT(Node root, Node p, Node q) {if (root == null || root == p || root == q) return root;
Node left/right = lcaBT (root.l, p, q);return left == null ? right : right == null ? left : root;}

}
private static void mergeArray(int A[], int m, int B[], int n) { // a[]=10,b[]=2,3 merge a[]=2,b=[3,10]
     for (int i=n-1; i>=0; i--){
           int j, e = A[m-1];
           for (j=m-2; j >= 0 && A[j] > B[i]; j--) A[j+1] = A[j];
           if (j != m-2 || e > B[i]){ A[j+1] = B[i]; B[i] = e;}
     }
}
  public int longestIncreasingContinuousSubsequence(int[] A) {
      int result = 1,len=1,n = A.length;
      for (int i = 1; i < n; i++) { // opposite do for right to left n-1 to 0
          if (A[i] > A[i - 1]) len++;//reverse condition A[i-1]>A[i]
          else len = 1;
          result = Math.max(len, result);
      }
      return ans;
  }
    private static int countSubarraysSumK(int[] A, int k) { // put k=0
        HashMap<Integer, Integer> hm = new HashMap<>();
        hm.put(0, 1); int res = 0,sum = 0,n=A.length;
        for(int i=0;i<n; i++){
            sum += A[i];
            int v = hm.getOrDefault(sum-k, 0);//if(map.containsKey(sum - k))result = Math.max(result, i - map.get(sum - k));
            res += v;//map.putIfAbsent(sum,i); return result; map.put(0,-1) for max size subarray sum k
            hm.put(sum, hm.getOrDefault(sum,0)+1);
        }
        return res;
    }
    private static double medianOfSortedArrays(int A[], int m, int B[], int n){
          int i=0,j=0,k=0; int[] C=new int[m+n], n=C.length;
          while (i<m && j <n) C[k++]= (A[i] < B[j])? A[i++]:B[j++];
          while (i<m) C[k++]=A[i++];
          while (j<n) C[k++]=B[j++];
          return (n%2==1)? C[n/2+1]: (C[n/2]+C[n/2+1])/2.0;
    }

    private static int findMinRotatedSortedArray(int[] A) {
        int i = 0, j = A.length - 1;
        while (i + 1 < j) {
            int mid = i + (j - i) / 2;//if target==nums[mid]return true for search
            if (A[mid] > A[j]) i = mid;//if target<nums[i]>nums[mid]i=mid else j=mid
            else if (A[mid] < A[j]) j = mid;//if target>nums[j]<nums[mid]j=mid else i=mid
            else j--;
        } //return (nums[i] or nums[j]==target)? true:false;
        return Math.min(A[i], A[j]);
    }


private static boolean canAttendAllMeetings(Interval[] intervals) {//Interval(int s, int e) { start = s; end = e; }
    Arrays.sort(intervals, (x, y) -> x.start - y.start);//Interval() { start = 0; end = 0; }
    for (int i = 1; i < intervals.length; i++) if(intervals[i-1].end > intervals[i].start) return false;
    return true;
}
public int minMeetingRooms(Interval[] intervals) {
        TreeMap<Integer, Integer> tmap = new TreeMap<>(); int result = 0, count = 0;
        for (Interval i : intervals) {
            int start = i.start, end = i.end;
            tmap.put(start, tmap.getOrDefault(start, 0) + 1);
            tmap.put(end, tmap.getOrDefault(end, 0) - 1);
        }
        for (int k : tmap.keySet()) {
            count += tmap.get(k);
            result = Math.max(result, count);
        }
        return result;
}
    public List<String> findItinerary(String[][] tickets, String start) {
        LinkedList<String> list = new LinkedList<>();​
        HashMap<String, PriorityQueue<String>> hm = new HashMap<>();
        for(String[] ticket : tickets){
            if(!hm.containsKey(ticket[0])) hm.put(ticket[0], new PriorityQueue<String>());​
            hm.get(ticket[0]).add(ticket[1]);
        }​
        dfs(list, start, map);​
        return new ArrayList<String>(list);
    }​
    private void dfs(LinkedList<String> list, String str, HashMap<String, PriorityQueue<String>> hm){
            while(hm.containsKey(str) && !hm.get(str).isEmpty()) dfs(list, hm.get(str).poll(), hm);
            list.offerFirst(airport);
    }
    private int[] productArrayExceptSelf(int[] nums) { // [1,2,3,4], return [24,12,8,6]
        int[] result = new int[nums.length]; int left=1,n=nums.length;
        result[nums.length-1]=1;
        for(int i=n-2; i>=0; i--) result[i]=result[i+1]*nums[i+1];
        for(int i=0; i<n; i++){
            result[i]=result[i]*left;
            left = left*nums[i];
        }
        return result;
    }

      public List<List<Integer>> subsetsDistinctArray(int[] nums) {
          List<List<Integer>> result = new ArrayList<>();
          result.add(new ArrayList<>());
          for(int n : nums){
              int size = result.size();
              for(int i=0; i<size; i++){
                List<Integer> subset = new ArrayList<>(result.get(i));
                subset.add(n);
                result.add(subset);
            }
        }
        return result;
    }
    private static double movingAverageNext(int val) {
          if(q.size() == size)sum = sum - q.poll(); // use queue q as linkedlist
          q.offer(val);//Moving(int s ){q=new LinkedList;size=s}
          sum += val;
          return sum/q.size();
    }
    private static int[] slidingWindowMaximum(int[] nums, int k) {  ​
             int n = nums.length; int[] sliding_max = new int[n - k + 1];​
             int[] max_left = new int[n]; int[] max_right = new int[n];
​             max_left[0] = nums[0];  max_right[n - 1] = nums[n - 1];
​             for (int i = 1; i < n; i++) {
                      max_left[i] = (i % k) == 0 ? nums[i] : Math.max(max_left[i - 1], nums[i]);
​                      int j = n - i - 1;
                      max_right[j] = (j % k == 0) ? nums[j] : Math.max(max_right[j + 1], nums[j]);
              }​
              for (int i = 0; i < sliding_max.length; i++) sliding_max[i] = Math.max(max_right[i], max_left[i + k - 1]);
              return sliding_max;
      }
      private static int[] slidingWindowMaximum(int[] nums, int k) { // [1, 2, 7, 7, 8] window k=3, return [7, 7, 8]
              int[] result = new int[nums.length-k+1]; int n=nums.length;
              LinkedList<Integer> dq = new LinkedList<Integer>();
              for(int i=0; i<n; i++){
                  if(!dq.isEmpty()&&dq.peekFirst()==i-k)dq.poll();
                  while(!dq.isEmpty()&&nums[dq.peekLast()]<nums[i])dq.removeLast();
                  dq.offer(i);
                  if(i+1>=k)result[i+1-k]=nums[dq.peek()];
              }
              return result;
      }

      private int[] warmerTemperatureWait(int[] A) { // T=[74,75,71,69,72,76,73],output [1,4,2,1,1,0,0].
            Deque <Integer> stack = new ArrayDeque<>(); int n=A.length;
            int[] result = new int[n];//nextgreaterelement of B in A,Map<I,I>,result[B.length]
            for (int i = 0; i < n; i++) {//nums1=[4,1,2] nums2 = [1,3,4,2]Output: [-1,3,-1]
                    while (!stack.isEmpty() && A[i] > A[stack.peek()]) {
                            int id = stack.pop();//map.put(A[id],A[i]);
                            result[id] = i - id;//CircularyArraynextgreater for(i0++<2*n)iRi%n result.fill-1,result[id]=A[i%n]
                    }
                    stack.push(i);
            }
            return result; //for(j0++<B)result[j]=map.getOrDefault(B[j],-1)
       }
       private static int largestRectangleAreaHistorgram(int[] nums) {
              LinkedList < Integer > stack = new LinkedList < > ();
              int result = 0, n=nums.length;​
              for (int i = 0; i <= n; i++) {
                        while (!stack.isEmpty() && (i == n || nums[stack.peek()] > nums[i])) {
                              int height = nums[stack.pop()];
                              int width = (!stack.isEmpty()) ? i - stack.peek() - 1 : i;
                              max = Math.max(max, height * width);
                        }​
                        stack.push(i);​
              }​
              return result;
        }
        public int maximalRectangle(char[][] matrix) {
              int[][] dp = new int[matrix.length][matrix[0].length]; int result = 0;
              for (int i = 0; i < dp.length; i++) {
                      for (int j = 0; j < dp[0].length; j++) {
                            dp[i][j] = matrix[i][j]-'0';
                            if (dp[i][j] > 0 && i>0) dp[i][j] += dp[i - 1][j];
                      }
                }
                for (int[] nums : dp) result=Math.max(largestRectangleAreaHistorgram(nums), result);​
                return result;
        }
        private static int carFleetReachingDestination(int target, int[] pos, int[] speed) {
                  TreeMap<Integer, Double> tm = new TreeMap<>(); int result = 0; double cur = 0;
                  for (int i = 0; i < pos.length; ++i) tm.put(-pos[i], (double)(target - pos[i]) / speed[i]);
                  for (double time : tm.values()) if (time > cur) {cur = time;result++;}
                  return result;//target=12,pos=[10,8,0,5,3]speed=[2,4,1,1,3]Output=3
        }
        private static List<String> topKFrequentWords(String[] nums, int k) {
              Map<String, Integer> hm = new HashMap();
              for (String num: nums) hm.put(word, hm.getOrDefault(word, 0) + 1);
              List<String> ls = new ArrayList(hm.keySet());
              Collections.sort(ls, (w1, w2) -> hm.get(w1).equals(hm.get(w2)) ? w1.compareTo(w2) : hm.get(w2) - hm.get(w1));
​              return ls.subList(0, k);
        }
        private static List<Integer> topKFrequentNums(int[] nums, int k) {
              Map<Integer,Integer> hm=new HashMap<>(); List<Integer> result=new ArrayList<>();
              for(int num:nums)  hm.put(num,hm.getOrDefault(num,0)+1);
              PriorityQueue<Integer> pq=new PriorityQueue<>((a,b)->hm.get(b)-hm.get(a));
              for(int key:hm.keySet()){
                    pq.offer(key);
                    if(pq.size()>hm.size()-k)result.add(pq.poll());
              }
              return result;
        }

    public static int fibReturnNth(int n) {
        if (n == 0) return 0;
        int prev = 0, res = 1, next;
        for (int i = 2; i <= n; i++) {
            next = prev + res;
            prev = res;
            res = next;
        }
        return res;
    }
    public static int maxContiguousSum(int arr[]) {
        int i, len = arr.length, cursum = 0, maxsum = Integer.MIN_VALUE;
        if (len == 0) return 0;
        for (i = 0; i < len; i++) {
            cursum += arr[i];
            if (cursum > maxsum) maxsum = cursum;
            if (cursum <= 0) cursum = 0;
        }
        return maxsum;
    }
    private static int cutRodBestPrice(int[] price, int n) { // rod length n
        int val[] = new int[n + 1]; val[0] = 0;
        for (int i = 1; i <= n; i++) {
            int max_val = Integer.MIN_VALUE;
            for (int j = 0; j < i; j++) max_val = Math.max(max_val, price[j] + val[i - j - 1]);
            val[i] = max_val;
        }
        return val[n];
    }
    private static void threeSumTriplet(int[] a, int target){
      Arrays.sort(a);	// Sort the array if array is not sorted
      for(int i=0;i<n;i++){
        int l=i+1,r=n-1;
        while(l<r){ //if you want all the triplets write l++;r--; insted of break;
          if(a[i]+a[l]+a[r]==target) {System.out.println(a[i]+" "+ a[l]+" "+a[r]);break;}
          else if(a[i]+a[l]+a[r]<target) l++;
          else r--;
        }
      }
    }
    private static int minTrialsEggDropping(int n, int m) { // n eggs & m floors
        int[][] eggFloor = new int[n + 1][m + 1]; int result, x;
        for (int i = 1; i <= n; i++) {
            eggFloor[i][0] = 0;   // Zero trial for zero floor.
            eggFloor[i][1] = 1;   // One trial for one floor
        } // j trials for only 1 egg
        for (int j = 1; j <= m; j++) eggFloor[1][j] = j;
        for (int i = 2; i <= n; i++) { // Using bottom-up approach in DP
            for (int j = 2; j <= m; j++) {
                eggFloor[i][j] = Integer.MAX_VALUE;
                for (x = 1; x <= j; x++) {
                    result = 1 + Math.max(eggFloor[i - 1][x - 1], eggFloor[i][j - x]);
                    if (result < eggFloor[i][j]) eggFloor[i][j] = result; // choose min of all values for particular x
                }
            }
        }
        return eggFloor[n][m];
    }
    public List < Integer > spiralOrderMatrix(int[][] matrix) { // if (matrix.length == 0) return ans;
        List result = new ArrayList();
        int r1 = 0, r2 = matrix.length - 1;
        int c1 = 0, c2 = matrix[0].length - 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; c++) result.add(matrix[r1][c]);
            for (int r = r1 + 1; r <= r2; r++) result.add(matrix[r][c2]);
            if (r1 < r2 && c1 < c2) {
                for (int c = c2 - 1; c > c1; c--) result.add(matrix[r2][c]);
                for (int r = r2; r > r1; r--) result.add(matrix[r][c1]);
            }
            r1++; r2--; c1++; c2--;
        }
        return ans;
    }
    public void setZeroMatrix(int[][] M) {
        int m = M.length, n = M[0].length; boolean row = false, col = false;
        for(j0++<n) if(matrix[0][j] == 0) { row = true; break; }
        for(i0++<m) if(matrix[i][0] == 0) { col = true; break;}
        for(i1++<m)for(j1++<n)if(M[i][j] == 0) {M[i][0] = 0;M[0][j] = 0;}
        for(i1++<m)for(j1++<n) if(M[i][0] == 0 || M[0][j] == 0) M[i][j] = 0;
        if(row) for(int j = 0; j < n; j++) M[0][j] = 0;//update first row
        if(col) for(int i = 0; i < m; i++) M[i][0] = 0;//update first col

    }
public int kthSmallestElementMatrix(int[][] mx, int k) {
    int m=mx.length, n=mx[0]length; int lo = mx[0][0], hi = mx[m-1][n - 1] + 1;
    while(lo < hi) {
        int mid = lo + (hi - lo) / 2, count = 0,  j = n - 1;
        for(int i = 0; i < m; i++) {
            while(j >= 0 && mx[i][j] > mid) j--;
            count += (j + 1);
        }
        if(count < k) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
    public void rotateMatrix90Degree(int[][] a) { // check if this working or not
            int n = a.length;
            for (int i = 0; i < n / 2; i++) {
                  for (int j = 0; j < n-i-1; j++) {
                        int temp = a[i][j];
                        a[i][j]=a[n-j-1][i];
                				a[n-j-1][i]=a[n-i-1][n-j-1];
                				a[n-i-1][n-j-1]=a[j][n-i-1];
                				a[j][n-i-1]=temp;
                  }
            }
    }

    private static double mincostToHireWorkers(int[] quality, int[] wage, int K) {
        int n = quality.length; Worker[] workers = new Worker[N];
        for (int i = 0; i < n; ++i) workers[i] = new Worker(quality[i], wage[i]);
        Arrays.sort(workers);double ans = Double.MAX_VALUE;int sumq = 0;
        PriorityQueue<Integer> pq = new PriorityQueue();
        for (Worker worker: workers) {
              pq.offer(-worker.quality);
              sumq += worker.quality;
              if (pq.size() > K) sumq += pq.poll();
              if (pq.size() == K) result = Math.min(result, sumq * worker.ratio());
        }​
        return result;
    }


    public int numIslands(boolean[][] grid) {
          int m = grid.length,n = grid[0].length, resul=0;
          for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                      if (grid[i][j] == false) continue;
                      nums++;
                      dfs(grid, i, j);
                }
          }
        return nums;
    }​
    private void dfs(boolean[][] grid, int i, int j) {
        if (i < 0 || i >= m || j < 0 || j >= n) return;
        if (grid[i][j] == true) { grid[i][j] = false; dfs(grid, i - 1, j); dfs(grid, i + 1, j);
            dfs(grid, i, j - 1); dfs(grid, i, j + 1);
        }
    }


  private static String removeDuplicate(String s) {// check if this works for unsorted array, Integer.toString()
      if (s == null || s.isEmpty()) return s;
      StringBuilder sb = new StringBuilder(); int n = s.length();
      for (int i = 0; i < n; i++) if (sb.toString().indexOf(s.charAt(i)) == -1) sb.append(String.valueOf(s.charAt(i)));
      return sb.toString();
  }
    private static boolean isValidParentheses(String s) {
            Stack<Character> stack = new Stack<>();
            for(char c : s.toCharArray()){
                  switch(c){
                        case ']': if(stack.isEmpty() || stack.pop()!='[') return false; break;
                        case ')': if(stack.isEmpty() || stack.pop()!='(') return false; break;
                        case '}': if(stack.isEmpty() || stack.pop()!='{') return false; break;
                        default: stack.push(c);
                  }
            }
            return stack.isEmpty();
    }
    private static int longestValidParenthesesLen(String s) {
              Stack<Integer> stack = new Stack<Integer>();
              int n = s.length(), result = 0; stack.push(-1);
              for (int i = 0; i < n; i++) {
                      if (s.charAt(i) == '(') stack.push(i);
                      else {
                            stack.pop();
                            if (stack.isEmpty()) stack.push(i);
                            else result = Math.max(result, i - stack.peek());
                        }
              }
              return result;
    }
    private static int shortestDistanceIndexDiff(String[] words, String word1, String word2) {
          int i1 = -1,i2 = -1,result = Integer.MAX_VALUE, n=words.length;
          for (int i = 0; i < n; i++) {
                if (words[i].equals(word1)) i1 = i;
                if (words[i].equals(word2)) i2 = i;
                if (i1 != -1 && i2 != -1) result = Math.min(Math.abs(i1 - i2), result);
          }
          return result;
    }
​    public static int minDistance(String word1, String word2) {//insert,replace,delete
        int len1 = word1.length(); int len2 = word2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
     	  for (int i = 0; i <= len1; i++) dp[i][0] = i;
        for (int j = 0; j <= len2; j++) dp[0][j] = j;
        for (int i = 0; i < len1; i++) {
            char c1 = word1.charAt(i);
            for (int j = 0; j < len2; j++) {
                char c2 = word2.charAt(j);
                if (c1 == c2) dp[i + 1][j + 1] = dp[i][j];
                else {
		                int replace = dp[i][j] + 1;
                    int insert = dp[i][j + 1] + 1;
                    int delete = dp[i + 1][j] + 1;
                    int min = replace > insert ? insert : replace;
                    min = delete > min ? min : delete;
                    dp[i + 1][j + 1] = min;
                }
            }
        }
        return dp[len1][len2];
    }

    private static int knapSack(int W, int wt[], int val[], int n) throws IllegalArgumentException {
        if(wt == null || val == null) throw new IllegalArgumentException();
        int i, w; int rv[][] = new int[n + 1][W + 1];
        for (i = 0; i <= n; i++) { // Build table rv[][] in bottom up manner
            for (w = 0; w <= W; w++) {
                if (i == 0 || w == 0) rv[i][w] = 0;
                else if (wt[i - 1] <= w) rv[i][w] = Math.max(val[i - 1] + rv[i - 1][w - wt[i - 1]], rv[i - 1][w]);
                else rv[i][w] = rv[i - 1][w];
            }
        }
        return rv[n][W];
    }
    public boolean isIsomorphic(String s, String t) { // if (s.length() != t.length()) return false;
          int[] p = new int[256]; int[] q = new int[256]; // Input: s = "egg", t = "add" Output: true
          Arrays.fill(p, -1); Arrays.fill(q, -1);​
          for (int i = 0; i < s.length(); i++) {
                if (p[s.charAt(i)] != q[t.charAt(i)]) return false;
                p[s.charAt(i)] = i;
                q[t.charAt(i)] = i;
          }​
          return true;
    }
      private static List<List<String>> groupAnagramList(String[] strs) { // if (strs.length == 0) return new ArrayList();
            Map<String, List> result = new HashMap<String, List>();
            for (String s : strs) {
                  char[] c = s.toCharArray(); Arrays.sort(c);
                  String key = String.valueOf(c);
                  if (!result.containsKey(key)) result.put(key, new ArrayList());
                  result.get(key).add(s);
            }
            return new ArrayList(result.values());
    }
public List<Integer> findAnagrams(String s1, String s2) {// if (s == null || s.length() == 0 || p == null || p.length() == 0) return list;
      List<Integer> result = new ArrayList<>(); int[] hash = new int[256];
      for (char c : s2.toCharArray()) hash[c]++;
      int p = 0, q = 0, c = s2.length();
      while (q < s1.length()) {
            if (hash[s1.charAt(q)] >= 1) c--;
            hash[s1.charAt(q)]--;
            q++;
            if (c == 0) result.add(p);
            if (q - p == s2.length() ) {​
                  if (hash[s1.charAt(s2)] >= 0) c++;
                  hash[s1.charAt(s2)]++;
                  p++;​
            }​
      }
      return result;
}
    private static String mostOccurredWord(String str) {
        String[] words = str.replaceAll("[^a-zA-Z\\s]", " ").toLowerCase().split("\\s+");​
        Map<String, Integer> hm = new HashMap<>();
        String result = ""; int c = 0;
        for (String word: words) {​
                hm.put(word, hm.getOrDefault(word, 0) + 1);
                if (hm.get(word) > c) {
                    c = hm.get(word);
                    result = word;
                }
        }
        return result;
    }
    private static String binaryStringSum(String a, String b) {
          StringBuilder sb = new StringBuilder();
          int i = a.length() - 1, j = b.length() -1, c = 0;
          while (i >= 0 || j >= 0) {
                int sum = c;
                if (j >= 0) sum += b.charAt(j--) - '0';
                if (i >= 0) sum += a.charAt(i--) - '0';
                sb.append(sum % 2);
                c = sum / 2;
          }
          if (c != 0) sb.append(c);
          return sb.reverse().toString();
    }
    public String longestCommonPrefix(String[] strs) { //   if (strs == null || strs.length == 0)   return "";
              StringBuilder sb = new StringBuilder();​ int s = Integer.MAX_VALUE;​
              for (String str : strs) s = Math.min(s, str.length());
              for (int i = 0; i < s; i++) {
                    char curr = strs[0].charAt(i);
                    for (String str : strs) if (str.charAt(i) != curr) return sb.toString();
                    sb.append(curr);
              }
              return sb.toString();
    }
    private static int firstOccurenceBInA(String A, String B) { // if(needle.equals("")) return 0;
          int m = A.length(),n=A.length;
          for(int i=0; i<=m-n; i++) if(A.substring(i,i+n).equals(B)) return i;
          return -1;
    }
}

    private static int distanceBetweenStrings(String a, String b) {
        int len_a = a.length() + 1; int len_b = b.length() + 1;
        int[][] distance_mat = new int[len_a][len_b];
        for (int i = 0; i < len_a; i++) distance_mat[i][0] = i;
        for (int j = 0; j < len_b; j++) distance_mat[0][j] = j;
        for (int i = 0; i < len_a; i++) {
            for (int j = 0; j < len_b; j++) {
                int cost;
                if (a.charAt(i) == b.charAt(j)) cost = 0;
                 else cost = 1;
                distance_mat[i][j] = Math.min(distance_mat[i - 1][j], distance_mat[i - 1][j - 1], distance_mat[i][j - 1]) + cost;
            }
        }
        return distance_mat[len_a - 1][len_b - 1];
    }
    private static int openLockNoOfSequence(String[] deadends, String target) {
            Set<String> begin = new HashSet<>(); Set<String> end = new HashSet<>();
            Set<String> deads = new HashSet<>(Arrays.asList(deadends));
            begin.add("0000"); end.add(target); int level = 0; Set<String> temp;
            while(!begin.isEmpty() && !end.isEmpty()) {
                  if (begin.size() > end.size()) { temp = begin; begin = end; end = temp;}
                  temp = new HashSet<>();
                  for(String s : begin) {
                        if(end.contains(s)) return level;
                        if(deads.contains(s)) continue;
                        deads.add(s);
                        StringBuilder sb = new StringBuilder(s);
                        for(int i = 0; i < 4; i ++) {
                              char c = sb.charAt(i);
                              String s1 = sb.substring(0, i) + (c == '9' ? 0 : c - '0' + 1) + sb.substring(i + 1);
                              String s2 = sb.substring(0, i) + (c == '0' ? 9 : c - '0' - 1) + sb.substring(i + 1);
                              if(!deads.contains(s1)) temp.add(s1);
                              if(!deads.contains(s2)) temp.add(s2);
                        }
                  }
                  level ++;
                  begin = temp;
            }
            return -1;
      }
      //single number-array every element appears thrice except one-return the single one
      public int singleNumber(int[] A) {
              int ones = 0, twos = 0, threes = 0;
              for (int i = 0; i < A.length; i++) {
                    twos |= ones & A[i];
                    ones ^= A[i];
                    threes = ones & twos;
                    ones &= ~threes;
                    twos &= ~threes;
              }
              return ones;
      }

      //get max binary gap,ex- 9’s binary form is 1001, the gap is 2
      public static int getGap(int N) {
              int max = 0; int count = -1; int r = 0;
              while (N > 0) {
                  // get right most bit & shift right
                  r = N & 1;
                  N = N >> 1;
                  if (0 == r && count >= 0) count++;
                  if (1 == r) {
                      max = count > max ? count : max;
                      count = 0;
                  }
             }
             return max;
      }
     // return the no 1s bits by taking an unsigned 32 bit integer
      public int hammingWeight(int n) {
              int count = 0;
              for(int i=1; i<33; i++){
                    if(getBit(n, i) == true) count++;
              }
              return count;
      }
      public boolean getBit(int n, int i){
              return (n & (1 << i)) != 0;
      }
     // bitwise AND of numbers range m to n included
      public int rangeBitwiseAnd(int m, int n) {
            while (n > m) n = n & n - 1;
            return m & n;
      }

}
class StackUsingLinkedlist {
​    // A linked list node
    private class Node {
​        int data; // integer data
        Node link; // reference variavle Node type
    }
    // create globle top reference variable
    Node top;
    // Constructor
    StackUsingLinkedlist()
    {
        this.top = null;
    }
​    // Utility function to add an element x in the stack
    public void push(int x) // insert at the beginning
    {
        // create new node temp and allocate memory
        Node temp = new Node();
​        // check if stack (heap) is full. Then inserting an
        // element would lead to stack overflow
        if (temp == null) {
            System.out.print("\nHeap Overflow");
            return;
        }
​        // initialize data into temp data field
        temp.data = x;
​        // put top reference into temp link
        temp.link = top;
​        // update top reference
        top = temp;
    }
​    // Utility function to check if the stack is empty or not
    public boolean isEmpty()
    {
        return top == null;
    }
​    // Utility function to return top element in a stack
    public int peek()
    {
        // check for empty stack
        if (!isEmpty()) {
            return top.data;
        }
        else {
            System.out.println("Stack is empty");
            return -1;
        }
    }
​    // Utility function to pop top element from the stack
    public void pop() // remove at the beginning
    {
        // check for stack underflow
        if (top == null) {
            System.out.print("\nStack Underflow");
            return;
        }
​        // update the top pointer to point to the next node
        top = (top).link;
    }
​    public void display()
    {
        // check for stack underflow
        if (top == null) {
            System.out.printf("\nStack Underflow");
            exit(1);
        }
        else {
            Node temp = top;
            while (temp != null) {
​                // print node data
                System.out.printf("%d->", temp.data);
​                // assign temp link to temp
                temp = temp.link;
            }
        }
    }
}
public ListNode mergeKLists(List<ListNode> lists) {
    if (lists.size() == 0) {
        return null;
    }
    return mergeHelper(lists, 0, lists.size() - 1);
}
​
private ListNode mergeHelper(List<ListNode> lists, int start, int end) {
    if (start == end) {
        return lists.get(start);
    }
​
    int mid = start + (end - start) / 2;
    ListNode left = mergeHelper(lists, start, mid);
    ListNode right = mergeHelper(lists, mid + 1, end);
    return mergeTwoLists(left, right);
}
}
