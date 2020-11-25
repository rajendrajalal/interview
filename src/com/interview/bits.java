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
    private static boolean isPalindrome(String s) {
        return (s == null || s.length() <= 1) || s.equals(new StringBuilder(s).reverse().toString());
    }
    private static boolean isPalindrome(String s){//check num binary representation is palindrome
          for (int i = 0, j = s.length() - 1; i < j; ++i, --j) if (s.charAt(i) != s.charAt(j)) return false;
          return true;
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
    public int maxSubArraySum(int[] A) {//for minSubarray replace min with max vice versa
        int M = Integer.MIN_VALUE, min = 0, prefixSum = 0, n=nums.length;
        for(int i = 0; i < n; i++){
            prefixSum += A[i];
            M = Math.max(M, prefixSum - min);
            min = Math.min(min, prefixSum);
        }
        return M;
    }
    private int minSubArrayLenSumEqualsK(int[] A, int k) { // [2,3,1,2,4,3] & s=7, the subarray [4,3] has min length of 2
        int n = nums.length, res = Integer.MAX_VALUE, f = 0,  sum = 0;
        for (int i = 0; i < n; i++) {
              sum += A[i];
              while (sum >= k) { res = Math.min(result, i + 1 - f); sum -= nums[f++];}
        }
        return res==Integer.MAX_VALUE? 0: res;
    }
    private static void mergeArrayInOne(int A[], int m, int B[], int n) { // m & n are no of elements
          int i = m - 1; int j = n - 1; int k = m + n - 1;
          while (k >= 0) A[k--] = (j < 0 || (i >= 0 && A[i] > B[j]))? A[i--]: B[j--];
    }
    private static Node mergeTwoList(Node A, Node B) { // m & n are no of elements
          if (A.v < B.v) { A.n = mergeTwoList(A.n, B);return A;}
          else {B.n = mergeTwoList(A, B.n); return B;}
    }
    private static Node oddEvenList(Node A) {
          if (A != null) {​ Node f = A, f1 = A.n, f2 = f1;
​               while (f1 != null && f1.n != null) {f.n = f.n.n; f1.next = f.n.n; f = f.n; f1 = f1.n;}
               f.n = f2;
           }
           return A;
    }
    private static Node getIntersectionNode(Node A, Node B) {
          Node p = A, q = B;
          while (p != q) {
                p =  (p != null) ? p.n : B;
                q =  (q != null) ? q.n : A;
          }
          return p;
    }
    private static Node reverseList(Node A) { //if (A == null || A.next == null) return A;
          Node p = reverseList(A.n);
          A.n.n = A; A.n = null;
          return p;
    }
    private static Node middleNode(Node A) {
       Node p = A, q = A;
       while (q != null && q.n != null) {p = p.n; q = q.n.n;}
       return p;
   }
   private static boolean isPalindromeList(Node A) {
        Node mid=middleNode(A); mid=reverseList(mid);
        while (A != null && mid != null) {
            if (A.v != mid.v) return false;
            A = A.n; mid = mid.n;
        }
        return true;
   }
   private static Node deleteDuplicateNode(Node A) {
        Node p = A;
        while (p != null && p.n != null) {if (p.n.v == p.v) p.n = p.n.n; else p = p.n;}
        return A;
    }
    private static Node removeElementVal(Node A, int v) {
          Node dummy = new Node(0),curr = A, prev = dummy; dummy.n = A;
          while (curr != null) {
                if (curr.val == val) prev.n = curr.n;
                else prev = prev.n;
                curr = curr.n;
          }
          return dummy.n;
    }
    private Node removeNthEndNode(Node A, int n) {
            Node p, q, dummy = new Node(0);
            dummy.n = A; p = A; q = dummy;
            for (int i = 0; i < n; i++) p = p.n;
            while (p != null) {p = p.n; q = q.n;}
            q.n = q.n.n;
            return dummy.n;
     }
     private static void mergeArray(int A[], int m, int B[], int n) { // a[]=10,b[]=2,3 merge a[]=2,b=[3,10]
          for (int i=n-1; i>=0; i--){
                int j, e = A[m-1];
                for (j=m-2; j >= 0 && A[j] > B[i]; j--) A[j+1] = A[j];
                if (j != m-2 || e > B[i]){ A[j+1] = B[i]; B[i] = e;}
          }
    }
    private static double medianOfSortedArrays(int A[], int m, int B[], int n){
          int i=0,j=0,k=0; int[] C=new int[m+n], n=C.length;
          while (i<m && j <n) C[k++]= (A[i] < B[j])? A[i++]:B[j++];
          while (i<m) C[k++]=A[i++];
          while (j<n) C[k++]=B[j++];
          return (n%2==1)? C[n/2+1]: (C[n/2]+C[n/2+1])/2.0;
    }
    private static int maxProductValueArray(int[] A) {
            int min = A[0],max = A[0], result = A[0], n=A.length;
            for (int i = 1; i < n; i++) {
                if (A[i] < 0) { int temp = max; max = min; min =temp;}
                max = Math.max(A[i], max * A[i]);
                min = Math.min(A[i], min * A[i]);
                result = Math.max(result, max);
            }
            return result;
    }
    private static int[] intersectionOfArrays(int[] A, int[] B) {
          Set<Integer> set = new HashSet<>(); Set<Integer> hs = new HashSet<>(); int m=A.length, n=B.length;
          for (int i = 0; i < m; i++) set.add(A[i]);
          for (int j = 0; j < n; i++) if (set.contains(B[j])) hs.add(B[j]);
          int[] result = new int[hs.size()]; int k = 0;
          for (Integer hsv : hs) result[k++] = hsv;
          return result;
    }
    private static int twoSumPairCountArray(int A[],int sum){
          int result=0, n=A.length
          for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (A[i] + A[j] == sum) result++;
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
    public int findPeakElementAnyIndex(int[] A) {
          int i = 0, j = A.length - 1;
          while (i < j) {
                int mid = i + (j - i) / 2;
                if (A[mid] > A[mid + 1]) j = mid;
                else i = mid + 1;
          }
          return i;
    }
    private static int peakIndexInArray(int[] A) {
          int i = 0; while (A[i] < A[i+1]) i++;
          return i;
    }
    public static int numJewelsInStones(String J, String S) { // J="aA", S="aAAbbbb" output 3
            int result=0;
            for(char c : S.toCharArray()) if(J.indexOf(c) != -1) result++;
            return result;
    }
    private static int singleNumber(int[] A) {​
            int result = 0;
            for (int Av : A) result ^= Av;
            return result;​
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
    private static void rotateStringByOffsetK(char[] str, int k) {
        int n= str.length; if (str == null || str.length == 0) return;​
        k = k % n;
        reverse(str, 0, n - k - 1); reverse(str, n - k, n - 1); reverse(str, 0, n - 1);
    }​
    private void reverse(char[] str, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            char temp = str[i]; str[i] = str[j]; str[j] = temp;
        }
    }
    private static String reverseWords(String s) { // input: take LeetCode output: LeetCode take
              String[] parts = s.trim().split("\\s+");
              String out = "";
              for (int i = parts.length - 1; i > 0; i--) out += parts[i] + " ";
              return out + parts[0];
    }
    private static String reverseWords(String s) {// input: take LeetCode output: ekat edoCteeL
            String[] str = s.split(" "); StringBuilder result = new StringBuilder();
            for (int i = 0; i < str.length; i++) str[i] = new StringBuilder(str[i]).reverse().toString();
            for (String st : str) result.append(st + " ");
            return result.toString().trim();
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
        private static  List<List<Integer>> levelBT(Node root) {
                List<List<Integer>> result = new ArrayList<List<Integer>>();
                level(root, 0, result);
                return result;
        }
        private static void level(Node root, int depth, List<List<Integer>> result) {
              if (depth == result.size()) result.add(new ArrayList<Integer>());
              result.get(depth).add(root.val);
              level(root.left, depth + 1, result);
              level(root.right, depth + 1, result);
​        }
        private static void in(Node root, List<Integer> result) {//if (root == null) return;
              in(root.left, result); result.add(root.val); in(root.right, result);
        }
        private static void post(Node root, List<Integer> result) {
              post(root.left, result); post(root.right, result); result.add(root.val);
        }
        private static void pre(Node root, List<Integer> result) {
              result.add(root.val); pre(root.left, result); pre(root.right, result);
        }
        private static int depth(Node root) { //if(root==null) return 0;
              return 1+Math.max(depth(root.left),depth(root.right));
        }
        private static boolean isSymmetric(TreeNode root) {
              return root == null || isMirror(root.left, root.right);
        }
        private static boolean isMirror(Node node1, Node node2) { //node1&node2 true,node1ornode false
              if (node1.val != node2.val) return false;
              return isMirror(node1.left, node2.right) && isMirror(node1.right, node2.left);
        }
        public boolean isBST(Node root) { // if (root == null) return true;
            return isValid(root, null, null);
        }
        private static boolean isValid(Node root, Integer min, Integer max) {// if (root == null) return true;
            if ((min != null && root.val <= min) ||  (max != null && root.val >= max)) return false;
            return isValid(root.left, min, root.val) && isValid(root.right, root.val, max);
       }
       public boolean hasPathSum(Node root, int sum) { // if (root == null) return false;
              if (root.val == sum && root.left == null && root.right == null) return true;
              return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
       }
       public int noOfPathSum(Node root, int sum) { // if (root == null) return 0;
              return path(root, sum) + noOfPathSum(root.left, sum) + noOfPathSum(root.right, sum);
        }
        private int path(Node root, int sum) { // if (root == null) return 0;
              return (root.val == sum ? 1 : 0) + path(root.left, sum - root.val) + path(root.right, sum - root.val);
        }
        private int maxPathSum(Node root) { // if (root == null) return 0;
              helper(root);
              return maxSum;
        }
        private static int helper(Node root) { // if (root == null) return 0;
            int leftPath = Math.max(helper(root.left), 0), rightPath = Math.max(helper(root.right), 0);​
            maxSum = Math.max(maxSum, root.val + leftPath + rightPath);
            return root.val + Math.max(leftPath, rightPath);
        }
        public boolean isSameTree(Node p, Node q) { //node1&node2 true,node1ornode false
              return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }
        private Node lcaBT(Node root, Node p, Node q) {
              if (root == null || root == p || root == q) return root;
              Node left = lcaBT (root.left, p, q);
              Node right = lcaBT (root.right, p, q);
              return left == null ? right : right == null ? left : root;
        }
        public int diameterBT(Node root) {
                depth(root);
                return max;
        }​
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
    private static int majorityElement(int[] nums) {//Arrays.sort(nums) return nums[nums.length/2]
        int count = 0, result = 0;
        for (int num : nums) {
            if (count == 0) result = num;
            count += (num == result) ? 1 : -1;
        }
        return result;
    }
    private static int pivotIndex(int[] nums) {// sum of left index equals to right index
        int sum = 0, left = 0,n=nums.length;
        for (int x: nums) sum += x;
        for (int i = 0; i < n; ++i) {
            if (left == sum - left - nums[i]) return i;
            left += nums[i];
        }
        return -1;
    }
    public int partitionArray(int[] nums, int k) {
        int offset = 0,temp=0, n= nums.length;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < k) {
                temp = nums[i];
                nums[i] = nums[offset];
                nums[offset] = temp;
                offset ++;
            }
        }
        return offset;
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
    public static int sumDivisors(int number) {//sum of all proper divisors of number
        int sum = 0;
        for (int i = 1, limit = number / 2; i <= limit; ++i) if (number % i == 0) sum += i;
        return sum;
    }
    public static long factorial(int n) {//factorial using recursion
        if (n < 0) throw new IllegalArgumentException("number is negative");
        return n == 0 || n == 1 ? 1 : n * factorial(n - 1);
    }
    public static int max(int[] array, int len) {//max element using recursion,for min replace max with min
        return len == 1 ? array[0] : Math.max(max(array, len - 1), array[len - 1]);
    }
    public static boolean isPerfectCube(int number) {
        int a = (int) Math.pow(number, 1.0 / 3);
        return a * a * a == number;
    }
    public static boolean checkIfPowerOfTwoOrNot(int number) {
        return number != 0 && ((number & (number - 1)) == 0);
    }
    public static boolean isPrime(int num) {
      for (int div = 3; div <= Math.sqrt(num); div += 2) if (num % div == 0) return false;
      return true;
    }
    public static int sumOfDigitsRecursion(int number) {
        number = number < 0 ? -number : number; /* calculate abs value */
        return number < 10 ? number : number % 10 + sumOfDigitsRecursion(number / 10);
    }
    public static int noOfCombinationsChange(int[] coins, int amount) {
        int[] combinations = new int[amount + 1]; combinations[0] = 1;
        for (int coin : coins) {
            for (int i = coin; i < amount + 1; i++) combinations[i] += combinations[i - coin];
        }
        return combinations[amount];
    }
    private void moveZeroes(int[] nums) {
          int count = 0; int n = nums.length;
          for (int i = 0; i < n; i++) if (arr[i] != 0) arr[count++] = arr[i];
          while (count < n) arr[count++] = 0;
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
    private static int longestConsecutiveSubsequenceLen(int[] nums){
            Set<Integer> set = new HashSet<>(); int result = 0;
            for(int i : nums) set.add(i);// [100, 4, 200, 1, 3, 2] return 4, subsequence [1, 2, 3, 4]
            for(int num : nums) {
                    int p = num - 1, q = num + 1;
                    while(set.remove(p)) p--;
                    while(set.remove(q)) q++;
                    ans = Math.max(ans,q - p - 1);
                    if(set.isEmpty()) return result;
            }
            return result;
    }
    private static int longestIncreasingSubsequenceLen(int[] nums){
          int i,j,result = 0,n=nums.length; int[] temp=new int[n]; Arrays.fill(temp,1);
          for (i = 1; i < n; i++) {
              for (j = 0; j < i; j++)
                  if (nums[i] > nums[j] && temp[i] < temp[j] + 1) temp[i] = temp[j] + 1;
          }
          for (i = 0; i < n; i++) if (max < temp[i]) max = temp[i];
          return max;
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
    private static int shortestDistanceIndexDiff(String[] words, String word1, String word2) {
          int i1 = -1,i2 = -1,result = Integer.MAX_VALUE, n=words.length;
          for (int i = 0; i < n; i++) {
                if (words[i].equals(word1)) i1 = i;
                if (words[i].equals(word2)) i2 = i;
                if (i1 != -1 && i2 != -1) result = Math.min(Math.abs(i1 - i2), result);
          }
          return result;
    }
    private static int distributeCandiesInTwo(int[] nums) {
          HashSet<Integer> hs = new HashSet<>();
          for (int num: nums) hs.add(num);
          return hs.size() < nums.length / 2 ? hs.size() : nums.length / 2;
    }
    private int minCandyDistribute(int[] ratings) { //atleast one to each child,higher rating get more than neighbours
        int[] candies = new int[ratings.length];  Arrays.fill(candies, 1); int n=ratings.length, result=0;
        for (int i=0; i<n-1; i++) { // forward
              if (ratings[i] >= ratings[i+1]) continue;
              candies[i+1] = candies[i] + 1;
        }
        for (int i=n-1; i>0; i--) { // backward
              if (ratings[i] >= ratings[i-1]) continue;
              candies[i-1] = Math.max(candies[i] + 1, candies[i-1]);
        }
        for (int i=0; i<candies.length; i++) result += candies[i];
        return result;
    }
    private static boolean canJump(int[] nums) {
          int n= nums.length, p = n - 1;
          for (int i = n - 1; i >= 0; i--) if (i + nums[i] >= p) p = i;
          return p == 0;
    }
    private static int minJump(int[] A) {
            int result = 0, p = 0, q = 0;
            for (int i = 0; i < A.length - 1; i++) {
                  p = Math.max(p, i + A[i]);
                  if (i == q) {jumps++; q = p; }
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
        private static int[] topKLargestNums(int[] nums, int k) {
                Arrays.sort(nums);int[] result = new int[k]; int j = 0, n=nums.length;
                for (int i = n - 1; i > n - k - 1; i--) {
                      result[j] = nums[i];
                      j++;
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
     private static int countSetBits(int num) {
        int cnt = 0;
        while (num != 0) {
            num = num & (num - 1);
            cnt++;
        }
        return cnt;
    }
    private static int countCharacters(String str) return str.replaceAll("\\s", "").length();
    private static int wordCount(String s) {
        if (s == null || s.isEmpty()) return 0;
        return s.trim().split("[\\s]+").length;
    }
    private static boolean containsDuplicate(int[] nums) { // if(nums==null || nums.length==0) return false;
            HashSet<Integer> set = new HashSet<Integer>();
            for(int i: nums)if(!set.add(i)) return true;
            return false;
    }
    private static String removeDuplicate(String s) {// check if this works for unsorted array, Integer.toString()
        if (s == null || s.isEmpty()) return s;
        StringBuilder sb = new StringBuilder(); int n = s.length();
        for (int i = 0; i < n; i++) if (sb.toString().indexOf(s.charAt(i)) == -1) sb.append(String.valueOf(s.charAt(i)));
        return sb.toString();
    }
    private static int removeDuplicates(int[] A) { // sorted array
        if (A.length < 2) return A.length;
        int j = 0; int i = 1;
        while (i < A.length) {
              if (A[i] != A[j]) {
                  j++;
                  A[j] = A[i];
              }
              i++;
        }
        return j + 1;
    }
private static int findKthLargest(int[] nums, int k) { // Arrays.sort(nums); return nums[n-k]
          PriorityQueue<Integer> pq = new PriorityQueue<>();
          for(int val : nums) {
                pq.offer(val);
                if(pq.size() > k) pq.poll();
          }
          return pq.peek();
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
    private static void sortArray3Objects(int[] nums) { // 0,1,2
        int[] buckets = new int[3],j=0;
        for(int num : nums) buckets[num]++;
        for(int val = 0; val < 3; val++) {
            for(int i = 0; count < buckets[val]; i++) nums[j++] = val;
        }
    }
    private static boolean isAlphabetical(String s) {
        s = s.toLowerCase();
        for (int i = 0; i < s.length() - 1; ++i) {
            if (!Character.isLetter(s.charAt(i)) || !(s.charAt(i) <= s.charAt(i + 1))) return false;
        }
        return true;
    }
    public static boolean isAnagrams(String s1, String s2) {
        s1 = s1.toLowerCase(); s2 = s2.toLowerCase();
        char[] values1 = s1.toCharArray(); char[] values2 = s2.toCharArray();
        Arrays.sort(values1); Arrays.sort(values2);
        return new String(values1).equals(new String(values2));
    }
    public boolean canPlaceFlowers(int[] flowerbed, int n) { // [1,0,0,0,1], n = 2 output true
        for (int i = 0; i < flowerbed.length; i++) {
            if (flowerbed[i] == 0 && (i == 0 || flowerbed[i - 1] == 0) &&
                (i == flowerbed.length - 1 || flowerbed[i + 1] == 0)) {
                    flowerbed[i] = 1; n--;
            }
            if (n <= 0) return true;
        }
        return false;
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
    public int minCoinChange(int[] coins, int amount) {
                    int[] dp = new int[amount+1];
                    Arrays.fill(dp, Integer.MAX_VALUE);
                    dp[0]=0;
                    for(int i=0; i<=amount; i++){
                            if(dp[i]==Integer.MAX_VALUE) continue;
                          for(int coin: coins){ //handle case when coin is Integer.MAX_VALUE
                                if(i<=amount-coin) dp[i+coin] = Math.min(dp[i]+1, dp[i+coin]);
                          }
                    }
                    return dp[amount]==Integer.MAX_VALUE ? -1: dp[amount];
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
      //best time to buy and sell stock atmost one transaction ie buy & sell,index is day & index value is price-return max profit
      public int maxProfit(int[] prices) {
              if(prices==null||prices.length<=1) return 0;
              int min=prices[0]; // min so far
              int result=0;
              for(int i=1; i<prices.length; i++){
                    result = Math.max(result, prices[i]-min);
                    min = Math.min(min, prices[i]);
              }
      return result;
      }
      //buy and sell stock multiple transaction but sell stock before buy,index is day & index value is price-return max profit
      public int maxProfit(int[] prices) {
              int profit = 0;
              for(int i=1; i<prices.length; i++){
                    int diff = prices[i]-prices[i-1];
                    if(diff > 0) profit += diff;
              }
      return profit;
      }
      //container with most water,ex- n non-negative integers a1, a2, ..., an, line i is at (i, ai) & (i,0)
      public int maxArea(int[] height) {
      				if (height == null || height.length < 2) {
      						return 0;
      				}
      				int max = 0;
      				int left = 0;
      				int right = height.length - 1;
      				while (left < right) {
      						max = Math.max(max, (right - left) * Math.min(height[left], height[right]));
      						if (height[left] < height[right]) left++;
      						else right--;
      				}
      return max;
      }
      //find path in 2d matrix move only down or right from top left to bottom right-return no of unique paths
      public int uniquePaths(int m, int n) {
                if(m==0 || n==0) return 0;
                if(m==1 || n==1) return 1;
                int[][] dp = new int[m][n];
                //left column
                for(int i=0; i<m; i++) dp[i][0] = 1;
                //top row
                for(int j=0; j<n; j++) dp[0][j] = 1;
                //fill up the dp table
                for(int i=1; i<m; i++){
                      for(int j=1; j<n; j++) dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
                return dp[m-1][n-1];
      }
      //house robber-returns the max amount of money can be robbed along street,adjacent houses cannot be robbed due to security
      public int rob(int[] nums) {
      				if(nums==null||nums.length==0) return 0;
      				if(nums.length==1) return nums[0];
      				int[] dp = new int[nums.length];
      				dp[0]=nums[0];
      				dp[1]=Math.max(nums[0], nums[1]);
      				for(int i=2; i<nums.length; i++) dp[i] = Math.max(dp[i-2]+nums[i], dp[i-1]);
      return dp[nums.length-1];
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
new repo- git init,undoing staged changes-git reset head
restore your changes-git checkout head, switch branch-git checkout branchname
orign live git remote v
git pull-git fetch and git merge(three way merge,diff from origin and merge)
merge conflict resolve-git reset --hard head
replacing set of commits-git rebase
undo-soft(reference back to specified commit only),mixed(soft()+staging area back),hard(mixed+rolls working directory)
stash(dirty state of working directory,saves stack of unfinished changes apply later)-switch to branches but don't want commit half done work
git stash apply, git cherry-pick commit_id
docker compose	up	starts	up	all	the	containers.
• docker compose	ps checks	the	status	of	the	containers	managed	by	docker compose.
• docker compose	logs outputs	colored	and	aggregated	logs	for	the	compose-managed
containers.
• docker compose	logs with	dash	f	option	outputs	appended	log	when	the	log	grows.
• docker compose	logs with	the	container	name	in	the	end	outputs	the	logs	of	a	specific
container.
• docker compose	stop stops	all	the	running	containers	without	removing	them.
• docker compose	rm removes	all	the	containers.
• docker compose	build rebuilds	all	the	images.
bridge-default ,none-isolated,host
dockermachine -multi host environment via vm
docker start relate to docker compose in docker swarm
design pattern-duck
interface quackable-quack()
xduck implement interface quackable
goose-honk(), adapter-gooseadapter implement quackable(goose, gooseadapter(goose),quack(goose.honk))
decorator-quackcounter implement quackable(quackable, quack(),quackcounter(quackable))-new quackcounter(new xduck())
factory-public abstract class AbstractDuckFactory {public abstract Quackable createMallardDuck()
  public class DuckFactory extends AbstractDuckFactory {
public Quackable createMallardDuck() {return new MallardDuck();}
public class CountingDuckFactory extends AbstractDuckFactory {public Quackable createMallardDuck() {
return new QuackCounter(new MallardDuck());}
AbstractDuckFactory duckFactory = new CountingDuckFactory();
Quackable mallardDuck = duckFactory.createMallardDuck();
manage,composite-public class Flock implements Quackable {
ArrayList quackers = new ArrayList();
public void add(Quackable quacker) {quackers.add(quacker);}
public void quack() {Iterator iterator = quackers.iterator()
  Flock flockOfDucks = new Flock();flockOfDucks.add(redheadDuck)
observer,individual duck behaviour
public interface QuackObservable {public void registerObserver(Observer observer);
public void notifyObservers();}
public interface Quackable extends QuackObservable {public void quack();}
public class Observable implements QuackObservable {ArrayList observers = new ArrayList();
QuackObservable duck;public Observable(QuackObservable duck) {this.duck = duck;}
public void registerObserver(Observer observer) {observers.add(observer);}
public void notifyObservers() {Iterator iterator = observers.iterator();
  public class MallardDuck implements Quackable {
Observable observable;public MallardDuck() {observable = new Observable(this);}
public void quack() {notifyObservers();}
public void registerObserver(Observer observer) {observable.registerObserver(observer);}
public void notifyObservers() {observable.notifyObservers();}
public interface Observer {public void update(QuackObservable duck);}
Quackologist quackologist = new Quackologist();
flockOfDucks.registerObserver(quackologist);
mvc pattern-
The Model consists of the data and business logic.
• The View is responsible for rendering the data in a way that is human readable on a
screen.
• The Controller is the brains behind the app and communicates with both the Model
and the View. The user interacts with the app via the Controller.
• When applying MVC to Android, the Android Activity ends up serving as both the
View and Controller, which is problematic for separation of concerns and unit
testing
mvp patern-
Unlike MVC wherein the main entry point is the Controller, in MVP the main entry
point is the View.
he Model is the data layer that handles business logic.
• The View displays the UI and informs the Presenter about user actions.
• The View extends Activity or Fragment.
• The Presenter tells the Model to update the data and tells the View to update the UI.
• The Presenter should not contain Android framework-specific classes.
• The Presenter and the View interact with each other through interfaces.
mvvm pattern-
View displays the UI and informs the other layers about user actions.
• ViewModel exposes information to the View.
Views display the UI and inform about user actions.
• The ViewModel gets the information from your Data Model, applies the necessary
operations and exposes the relevant data to your Views.
• The ViewModel exposes backend events to the Views so they can react accordingly.
• The Model, also known as the DataModel, retrieves information from your backend
and makes it available to your ViewModels.
• Model retrieves information from your datasource and exposes it to the ViewModels
viper pattern-relates to mvp
The View displays the User Interface.
• The Interactor performs actions related to the Entities.
• The Presenter acts as a command center to manage your Views, Interactors and
Routers.
• The Entity represents data in your app.
• The Router handles the navigation between your Views.
• VIPER is a great architecture pattern for projects that are expected to scale fast but
might be overkill for simple apps.
State In memory? Visible to user? In foreground?
nonexistent no no no
stopped yes no no
paused yes yes/partially* no
resumed yes yes yes
recyclerview-
The adapter is responsible for:
• creating the necessary ViewHolders when asked
• binding ViewHolders to data from the model layer when asked
The recycler view is responsible for:
• asking the adapter to create a new ViewHolder
• asking the adapter to bind a ViewHolder to the item from the backing data at a given position
Adapter.onCreateViewHolder(…) is responsible for creating a view to display, wrapping the view
in a view holder, and returning the result. In this case, you inflate list_item_view.xml and pass the
resulting view to a new instance of CrimeHolder.
Adapter.onBindViewHolder(holder: CrimeHolder, position: Int) is responsible for populating
a given holder with the crime from a given position. In this case, you get the crime from the crime
list at the requested position. You then use the title and data from that crime to set the text in the
corresponding text views.
When the recycler view needs to know how many items are in the data set backing it (such as when
the recycler view first spins up), it will ask its adapter by calling Adapter.getItemCount(). Here,
getItemCount() returns the number of items in the list of crimes to answer the recycler view’s request
The RecyclerView calls the adapter’s onCreateViewHolder(ViewGroup, Int) function to create a
new ViewHolder, along with its juicy payload: a View to display. The ViewHolder (and its itemView)
that the adapter creates and hands back to the RecyclerView has not yet been populated with data from
a specific item in the data set.
Next, the RecyclerView calls onBindViewHolder(ViewHolder, Int), passing a ViewHolder into this
function along with the position. The adapter will look up the model data for that position and bind it
to the ViewHolder’s View. To bind it, the adapter fills in the View to reflect the data in the model object.
After this process is complete, RecyclerView will place a list item on the screen.
activity-visible-onstart-onstop,foreground-onresume-onpause
The activity gets launched, and the
onCreate() method runs.
Any activity initialization code in the
onCreate() method runs. At this point, the
activity isn’t yet visible, as no call to onStart()
has been made.
The onStart() method runs. It gets
called when the activity is about to
become visible.
After the onStart() method has run, the user
can see the activity on the screen
The onResume() method runs. It gets
called when the activity is about to move
into the foreground.
After the onResume() method has run, the
activity has the focus and the user can interact
with it.
The onPause() method runs when the
activity stops being in the foreground.
After the onPause() method has run, the
activity is still visible but doesn’t have the focus.
If the activity moves into the
foreground again, the onResume()
method gets called.
The activity may go through this cycle many
times if the activity repeatedly loses and then
regains the focus.
The onStop() method runs when the
activity stops being visible to the user.
After the onStop() method has run, the activity
is no longer visible.
If the activity becomes visible to the
user again, the onRestart() method gets
called followed by onStart().
The activity may go through this cycle many
times if the activity repeatedly becomes invisible
and then visible again
Finally, the activity is destroyed.
The onStop() method will get called before
onDestroy().
