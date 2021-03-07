  SELECT DATEDIFF(year, '2011-10-02', '2012-01-01');
  SELECT DATE_PART('year', '2012-01-01'::date) - DATE_PART('year', '2011-10-02'::date);
supervised/task, unpervised/data driven, reinforcement/error
waterfall/sequence type- develop and then test, agile-continuous integration
timsort- merge sort and insertion sort
backpropagation(dl)-messenger telling network whether or not the net made a mistake when making prediction
GET carries request parameter appended in URL
POST carries request parameter in message body which makes it more secure way of transferring data from client t
cookies are employed to store user choices such as browsing session to trace the user preferences
Cache store online page resources during a browser for the long run purpose or to decrease the loading time
count even substring each contains even characters each type
datalinklayer(mac,llc)
bash script output-13,5,16,8
n queen-0(N)
docker-image rm(remove one or more images),images(list all images & sizes),rm(remove container)
kubectl
unrolledlinked list-linear search faster,less memory overhead & storage space,insertion etc faster
program output-0,1
binomial heap-set of binomial trees,all true
program output-1,1,2,2,5,5,7
bit stuffin in can protocol-5
average waiting time first come first serve-wrong option
dockerfile-expose
algo-merge,tim,cube
program output-return f,return new,return create
kubernetes-both
first come first serve
m colouring-check all edges, at last after visiting all nodes return true, 3,1,2,4,5
sed,permutation incomplete
# your code goes here
from collections import defaultdict 

lis = [5,1,1,3,5]
d={}
res =[-1]*len(lis)
#d =defaultdict(list)
for i,v in enumerate(lis):
    d.setdefault(v,[]).append(i)
#   d[v].append(i)
print(d)
for value,index in d.items():
    if (len(index)>1):
        index.sort()
        for i,v in enumerate(index):
            left=math.inf
            right=math.inf
            if(i==0):
                print(abs(index[i+1]-index[i]))
                res[v]=abs(index[i+1]-index[i])
            if(i>1 and i<len(index)):
                left=abs(index[i]-index[i-1])
                print(left)
                right = abs(index[i]-index[i+1])
                print(right)
                _min = min(left,right)
                res[v] = _min
            if(i==len(index)-1):
                res[v] = abs(index[i]-index[i-1])     
            
    
    
print(' '.join(map(str,res)))


from collections import Counter
def checkString(s):
    frequency = Counter(s)
    for i in frequency:
        if (frequency[i] % 2 == 1):
            return False
    return True
def subString(s, n):    
    count=0;
    for i in range(n):
        for j in range(i+1,n+1):
            size=j-i
            if(size%2==0 and checkString(s[i:j])):
                count+=1
    return count
or
def subString(s, n):
    # Pick starting point in outer loop
    # and lengths of different strings for
    # a given starting point
    res = [s[i:j] for i, j in combinations( 
            range(n + 1), r = 2) if((j-i)%2==0 and checkString(s[i:j]))]
    return len(res)
def divideArray(arr, n, k): 
      
    # Dp to store the values 
    dp = [[0 for i in range(n*k)]  
             for i in range(n*k)] 
  
    k -= 1
  
    # Fill up the dp table 
    for i in range(n - 1, -1, -1): 
        for j in range(0, k + 1): 
              
            # Intitilize maximum value 
            dp[i][j] = 10**9
  
            # Max element and the summ 
            max_ = -1
            min_=-1
            summ = 0
  
            # Run a loop from i to n 
            for l in range(i, n): 
                  
                # Find the maximum number 
                # from i to l and the summ 
                # from i to l 
                max_ = max(max_, arr[l])
                min_ = min(min_,arr[l])
                summ += arr[l] 
  
                # Find the summ of difference 
                # of every element with the 
                # maximum element 
                diff = max_-min_ 
  
                # If the array can be divided 
                if (j > 0): 
                    dp[i][j]= min(dp[i][j], diff + 
                                  dp[l + 1][j - 1]) 
                else: 
                    dp[i][j] = diff 
  
    # Returns the minimum summ 
    # in K parts 
    return dp[0][k]

def maxDegreeOfFreedom(s,n):
    res=-1
    for i in range(n-2):
        res = max(divideArray(s,n,i+2),res)
    return res-1
    

factorial
r = 1
    while n > 0:
        r = r * n
        n = n - 1
    return r
 or
  return 1 if n <= 0 else n * factorial(n - 1)
insertion sort
for i in range(1, len(arr)):   
        key = arr[i]   
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j] 
                j -= 1
        arr[j+1] = key 
heap sort 
 n = len(unsorted)
    for i in range(n // 2 - 1, -1, -1):
        heapify(unsorted, i, n)
    for i in range(n - 1, 0, -1):
        unsorted[0], unsorted[i] = unsorted[i], unsorted[0]
        heapify(unsorted, 0, i)
    return unsorted
def heapify(unsorted, index, heap_size):
    largest = index
    left_index = 2 * index + 1
    right_index = 2 * index + 2
    if left_index < heap_size and unsorted[left_index] > unsorted[largest]:
        largest = left_index

    if right_index < heap_size and unsorted[right_index] > unsorted[largest]:
        largest = right_index

    if largest != index:
        unsorted[largest], unsorted[index] = unsorted[index], unsorted[largest]
        heapify(unsorted, largest, heap_size)
merge sort
if len(l) < 2: return
    pivot = len(l)//2
    left, right = l[:pivot], l[pivot:]
    merge_sort(left)
    merge_sort(right)
    k = 0
    while len(left) > 0 and len(right) > 0:
        if left[0] < right[0]:
            l[k] = left.pop(0)
        else:
            l[k] = right.pop(0)
        k += 1

    rest = left + right
    while len(rest) > 0:
        l[k] = rest.pop(0)
        k += 1 
quick sort
if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
reverse string
if len(string) == 0: 
        return      
    temp = string[0] 
    reverse(string[1:]) 
    print(temp, end='') 
palindrome
s = s.upper()
    # Checking if both string are equal or not
    if (s == reverse(s)):
        return True
    return False
recursive palindrome
string = string.lower()
  if(len(string) > 2):
    if(string[0] == string[-1]):
      return checkString(string[1 : -1])
    else:
      return False
  else: return string[0] == string[-1]
reverse each word
st = list()  
    for i in range(len(string)): 
        if string[i] != " ": 
            st.append(string[i])
        else: 
            while len(st) > 0: 
                print(st[-1], end= "") 
                st.pop() 
            print(end = " ") 
    while len(st) > 0: 
        print(st[-1], end = "") 
        st.pop() 
reverse word in sentence
def reverse_word(s, start, end):
    while start < end:
        s[start], s[end] = s[end], s[start]
        start = start + 1
        end -= 1 
s = "i like this program very much"-o/p-much very program this like i
s = list(s)
start = 0
while True:
     try:
        end = s.index(' ', start) 
        reverse_word(s, start, end - 1)ble
        start = end + 1 
    except ValueError:
        reverse_word(s, start, len(s) - 1)
        break 
# Reverse the entire list
s.reverse()
s = "".join(s)
longest substring alphabetical
count = 0
    maxcount = 0
    result = 0
    for char in range(len(s) - 1):
        if (s[char] <= s[char + 1]):
            count += 1
            if count > maxcount:
                maxcount = count
                result = char + 1
        else:
            count = 0
    startposition = result - maxcount
    return startposition, result
linked list 
class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
def middle(self):
        fast = self.head
        slow = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        print slow.data
  def reverse(self,cur):
        cur = cur
        n = cur.next

        if n.next:
            self.reverse(cur.next)
            n.next = cur
            cur.next = None
        else:
            n.next = cur
            self.head = n
 ef getPairsCount(arr, n, sum): 
    m = [0] * 1000 
    # Store counts of all elements in map m
    for i in range(0, n):
        m[arr[i]] += 1 
    twice_count = 0 
    # Iterate through each element and increment
    # the count (Notice that every pair is
    # counted twice)
    for i in range(0, n): 
        twice_count += m[sum - arr[i]] 
        # if (arr[i], arr[i]) pair satisfies the
        # condition, then we need to ensure that
        # the count is  decreased by one such
        # that the (arr[i], arr[i]) pair is not
        # considered
        if (sum - arr[i] == arr[i]):
            twice_count -= 1 
    # return the half of twice_count
    return int(twice_count / 2)
 or
 def getPairsCount(arr, n, sum): 
    count = 0  # Initialize result
    for i in range(0, n):
        for j in range(i + 1, n):
            if arr[i] + arr[j] == sum:
                count += 1 
    return count
 coin change(v,items)
 ans = 99
    if v <= 0:
        return 0;
    for item in items:
        if v - item >= 0:
            update = 1 + coin_change(v - item, items)
            ans = min(ans, update);
    return ans;
def minimum_coins(value, denominations):
    result = []
    # Assuming denominations is sorted in descendig order
    for cur_denom in denominations:
        while cur_denom <= value:
            result.append(cur_denom)
            value = value - cur_denom

    return result
merge array into another
def mergeArrays(arr1, arr2, n1, n2): 
    arr3 = [None] * (n1 + n2) 
    i = 0
    j = 0
    k = 0
    while i < n1 and j < n2: 
        if arr1[i] < arr2[j]: 
            arr3[k] = arr1[i] 
            k = k + 1
            i = i + 1
        else: 
            arr3[k] = arr2[j] 
            k = k + 1
            j = j + 1
    while i < n1: 
        arr3[k] = arr1[i]; 
        k = k + 1
        i = i + 1
    while j < n2: 
        arr3[k] = arr2[j]; 
        k = k + 1
        j = j + 1
missing no
    x1 = a[0]
    x2 = 1     
    for i in range(1, n):
        x1 = x1 ^ a[i]         
    for i in range(2, n + 2):
        x2 = x2 ^ i     
    return x1 ^ x2
    or
    i, total = 0, 1 
    for i in range(2, n + 2):
        total += i
        total -= a[i - 2]
    return total
Largest Sum Contiguous Subarray
def maxSubArraySum(a,size):       
    max_so_far = 0
    max_ending_here = 0      
    for i in range(0, size): 
        max_ending_here = max_ending_here + a[i] 
        if max_ending_here < 0: 
            max_ending_here = 0
        elif (max_so_far < max_ending_here): 
            max_so_far = max_ending_here              
    return max_so_far
subarray sum exist 
    Map = {}  
    curr_sum = 0     
    for i in range(0,n):
        curr_sum = curr_sum + arr[i]      
        if curr_sum == Sum:0toi found
            return  
        if (curr_sum - Sum) in Map:
            print("Sum found between indexes", \ 
                   Map[curr_sum - Sum] + 1, "to", i)              
            return    
        Map[curr_sum] = i  
or
    curr_sum = arr[0] 
    start = 0    
    i = 1
    while i <= n: 
        while curr_sum > sum and start < i-1:           
            curr_sum = curr_sum - arr[start] 
            start += 1     
        if curr_sum == sum: 
            print ("Sum found between indexes") 
            print ("% d and % d"%(start, i-1)) 
            return 1
        if i < n: 
            curr_sum = curr_sum + arr[i] 
        i += 1
LCS problem 
def lcs(X , Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in xrange(m+1)] 
  
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 
LIS problem 
def lis(arr): 
    n = len(arr)   
    # Declare the list (array) for LIS and initialize LIS 
    # values for all indexes 
    lis = [1]*n   
    # Compute optimized LIS values in bottom up manner 
    for i in range (1 , n): 
        for j in range(0 , i): 
            if arr[i] > arr[j] and lis[i]< lis[j] + 1 : 
                lis[i] = lis[j]+1  
    # Initialize maximum to 0 to get the maximum of all 
    # LIS 
    maximum = 0  
    # Pick maximum of all LIS values 
    for i in range(n): 
        maximum = max(maximum , lis[i])   
    return maximum
rod cutting
def rod_cutting(price):    
    length = len(price)
    opt_price = [0] * (length + 1)

    for i in range(1, length + 1):
        opt_price[i] = max(
                    [-1] + [price[j] + opt_price[i - j - 1] for j in range(i)])
    return opt_price[length]
def dijkstra(graph, source):

    vertices, edges = graph
    dist = dict()
    previous = dict()

    for vertex in vertices:
        dist[vertex] = float("inf")
        previous[vertex] = None

    dist[source] = 0
    Q = set(vertices)

    while len(Q) > 0:
        u = minimum_distance(dist, Q)
        print('Currently considering', u, 'with a distance of', dist[u])
        Q.remove(u)

        if dist[u] == float('inf'):
            break

        n = get_neighbours(graph, u)
        for vertex in n:
            alt = dist[u] + dist_between(graph, u, vertex)
            if alt < dist[vertex]:
                dist[vertex] = alt
                previous[vertex] = u

    return previous
def zigzag_iterative(root: Node):
    if root == None: 
        return 
    s1 = [] # For levels to be printed from right to left  
    s2 = [] # For levels to be printed from left to right 
    s1.append(root)  
    while not len(s1) == 0 or not len(s2) == 0:           
        # Print nodes of current level from s1 and append nodes of next level to s2  
        while not len(s1) == 0: 
            temp = s1[-1]  
            s1.pop()  
            print(temp.data, end = " ")    
            # Note that is left is appended before right    
            if temp.left: 
                s2.append(temp.left)
            if temp.right:  
                s2.append(temp.right)  
        # Print nodes of current level from s2 and append nodes of next level to s1  
        while not len(s2) == 0: 
            temp = s2[-1]  
            s2.pop()  
            print(temp.data, end = " ")    
            # Note that is rightt is appended before left   
            if temp.right:  
                s1.append(temp.right)
            if temp.left: 
                s1.append(temp.left)  
# classic implementation of Singleton Design pattern 
class Singleton:   
    __shared_instance = 'GeeksforGeeks'  
    @staticmethod
    def getInstance():   
        """Static Access Method"""
        if Singleton.__shared_instance == 'GeeksforGeeks': 
            Singleton() 
        return Singleton.__shared_instance 
  
    def __init__(self):   
        """virtual private constructor"""
        if Singleton.__shared_instance != 'GeeksforGeeks': 
            raise Exception ("This class is a singleton class !") 
        else: 
            Singleton.__shared_instance = self

# Double Checked Locking singleton pattern    
import threading   
class SingletonDoubleChecked(object):   
    # resources shared by each and every 
    # instance   
    __singleton_lock = threading.Lock() 
    __singleton_instance = None  
    # define the classmethod 
    @classmethod
    def instance(cls):   
        # check for the singleton instance 
        if not cls.__singleton_instance: 
            with cls.__singleton_lock: 
                if not cls.__singleton_instance: 
                    cls.__singleton_instance = cls()   
        # return the singleton instance 
        return cls.__singleton_instance 



//python algo
// breath first search-shortest path,graph contains only segment = {} graph[“you”] = [“alice”, “bob”, “claire”]
//shortest path including cost-graph[“start”] = {},graph[“start”][“a”] = 6(neighbour hash),
costs = {},costs[“a”] = 6,parents = {},parents[“a”] = “start”
//greedy-min stations to cover all cities-stations = {},stations[“kone”] = 
set([“id”, “nv”, “ut”]),stations[“ktwo”] = set([“wa”, “id”, “mt”])
//dynamic programming-longest common substring
//dynamic programming-longest common subsequence
//remove duplicates and sorted
//square each odd number in lists
//sort tuple-name,age,score
//frequence of words in text
// filter list even numbers
//generat square of list via map function
//use map & filter to generate square of even numbers
---emailAddress
//fibonacci
//remove 0.4.5th items
----print(list(itertools.permutations([1,2,3])))
-----coin change
maxcapacity-weight,val/cutrod/staircaseways,minjumps,maxsubarraysum
split_coinspartitionmindifference/max_profitstock/max_area_containermostwater
count_tripletssumk/ missing_number/peakelementindex/kthsmallestnumer
longest increasing subarraylen/majority element/min swaps to sort
reverse array/rotate array/find subrray index/bfs/iscyclic/dfs
topological sort/middle list/reverse list/is palindrome/

# Python program to print all permutations with 
# duplicates allowed  
def toString(List): 
    return ''.join(List) 
def permute(a, l, r): 
    if l==r: 
        print toString(a) 
    else: 
        for i in xrange(l,r+1): 
            a[l], a[i] = a[i], a[l] 
            permute(a, l+1, r) 
            a[l], a[i] = a[i], a[l] # backtrack  
# Driver program to test the above function 
string = "ABC"
n = len(string) 
a = list(string) 
permute(a, 0, n-1) 

# Python3 program to find the length of the longest substring 
# without repeating characters 
NO_OF_CHARS = 256
  
def longestUniqueSubsttr(string):   
    # Initialize the last index array as -1, -1 is used to store 
    # last index of every character 
    lastIndex = [-1] * NO_OF_CHARS   
    n = len(string) 
    res = 0   # Result 
    i = 0  
    for j in range(0, n): 
        # Find the last index of str[j] 
        # Update i (starting index of current window) 
        # as maximum of current value of i and last 
        # index plus 1 
        i = max(i, lastIndex[ord(string[j])] + 1);   
        # Update result if we get a larger window 
        res =  max(res, j - i + 1)   
        # Update last index of j. 
        lastIndex[ord(string[j])] = j;   
    return res 

def longestUniqueSubsttr(string):        
    # Creating a set to store the last positions of occurrence 
    seen = {} 
    maximum_length = 0   
    # starting the inital point of window to index 0 
    start = 0        
    for end in range(len(string)):   
        # Checking if we have already seen the element or not 
        if string[end] in seen:   
            # If we have seen the number, move the start pointer 
            # to position after the last occurrence 
            start = max(start, seen[string[end]] + 1)    
        # Updating the last seen value of the character 
        seen[string[end]] = end 
        maximum_length = max(maximum_length, end-start + 1) 
    return maximum_length 

# Python 3 Program to find the longest common prefix    
# A Function to find the string having the minimum 
# length and returns that length 
def findMinLength(arr, n):   
    min = len(arr[0]) 
   
    for i in range(1,n): 
        if (len(arr[i])< min): 
            min = len(arr[i]) 
   
    return(min) 
   
# A Function that returns the longest common prefix 
# from the array of strings 
def commonPrefix(arr, n):   
    minlen = findMinLength(arr, n) 
    result ="" 
    for i in range(minlen):       
        # Current character (must be same 
        # in all strings to be a part of 
        # result) 
        current = arr[0][i]    
        for j in range(1,n): 
            if (arr[j][i] != current): 
                return result    
        # Append to result 
        result = result+current    
    return (result) 

# A Python3 Program to find the longest common prefix  
  
# A Utility Function to find the common  
# prefix between strings- str1 and str2  
def commonPrefixUtil(str1, str2):    
    result = ""  
    n1, n2 = len(str1), len(str2) 
    i, j = 0, 0  
    while i <= n1 - 1 and j <= n2 - 1:        
        if str1[i] != str2[j]:  
            break
        result += str1[i] 
        i, j = i + 1, j + 1      
    return result 
  
# A Divide and Conquer based function to  
# find the longest common prefix. This is  
# similar to the merge sort technique  
def commonPrefix(arr, low, high):    
    if low == high: 
        return arr[low]   
    if high > low:       
        # Same as (low + high)/2, but avoids  
        # overflow for large low and high  
        mid = low + (high - low) // 2  
        str1 = commonPrefix(arr, low, mid)  
        str2 = commonPrefix(arr, mid + 1, high)   
        return commonPrefixUtil(str1, str2)  

# A Dynamic Programming based Python3 program to 
# find minimum of coins to make a given change V
import sys  
# m is size of coins array (number of 
# different coins)
def minCoins(coins, m, V):     
    # table[i] will be storing the minimum 
    # number of coins required for i value. 
    # So table[V] will have result
    table = [0 for i in range(V + 1)] 
    # Base case (If given value V is 0)
    table[0] = 0 
    # Initialize all table values as Infinite
    for i in range(1, V + 1):
        table[i] = sys.maxsize 
    # Compute minimum coins required 
    # for all values from 1 to V
    for i in range(1, V + 1):         
        # Go through all coins smaller than i
        for j in range(m):
            if (coins[j] <= i):
                sub_res = table[i - coins[j]]
                if (sub_res != sys.maxsize and
                    sub_res + 1 < table[i]):
                    table[i] = sub_res + 1
    return table[V]
# Dynamic Programming Python implementation of Coin  
# Change problem 
def count(S, m, n): 
    # We need n+1 rows as the table is constructed  
    # in bottom up manner using the base case 0 value 
    # case (n = 0) 
    table = [[0 for x in range(m)] for x in range(n+1)]   
    # Fill the entries for 0 value case (n = 0) 
    for i in range(m): 
        table[0][i] = 1  
    # Fill rest of the table entries in bottom up manner 
    for i in range(1, n+1): 
        for j in range(m):   
            # Count of solutions including S[j] 
            x = table[i - S[j]][j] if i-S[j] >= 0 else 0  
            # Count of solutions excluding S[j] 
            y = table[i][j-1] if j >= 1 else 0  
            # total count 
            table[i][j] = x + y   
    return table[n][m-1] 

# Dynamic Programming bsed Python  
# implementation of Maximum Sum  
# Increasing Subsequence (MSIS) 
# problem 
  
# maxSumIS() returns the maximum  
# sum of increasing subsequence  
# in arr[] of size n 
def maxSumIS(arr, n): 
    max = 0
    msis = [0 for x in range(n)]   
    # Initialize msis values 
    # for all indexes 
    for i in range(n): 
        msis[i] = arr[i]   
    # Compute maximum sum  
    # values in bottom up manner 
    for i in range(1, n): 
        for j in range(i): 
            if (arr[i] > arr[j] and
                msis[i] < msis[j] + arr[i]): 
                msis[i] = msis[j] + arr[i]   
    # Pick maximum of 
    # all msis values 
    for i in range(n): 
        if max < msis[i]: 
            max = msis[i]   
    return max

# Dynamic Programming implementation of LCS problem 
  
def lcs(X , Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y)   
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in xrange(m+1)]   
    """Following steps build L[m+1][n+1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 

# Dynamic programming Python implementation 
# of LIS problem   
# lis returns length of the longest  
# increasing subsequence in arr of size n 
def lis(arr): 
    n = len(arr)   
    # Declare the list (array) for LIS and  
    # initialize LIS values for all indexes 
    lis = [1]*n   
    # Compute optimized LIS values in bottom up manner 
    for i in range (1 , n): 
        for j in range(0 , i): 
            if arr[i] > arr[j] and lis[i]< lis[j] + 1 : 
                lis[i] = lis[j]+1  
    # Initialize maximum to 0 to get  
    # the maximum of all LIS 
    maximum = 0  
    # Pick maximum of all LIS values 
    for i in range(n): 
        maximum = max(maximum , lis[i])   
    return maximum 

# Python3 program using to find length of
# the longest common substring recursion
 
# Returns length of function for longest
# common substring of X[0..m-1] and Y[0..n-1]
 
def lcs(i, j, count): 
    if (i == 0 or j == 0):
        return count 
    if (X[i - 1] == Y[j - 1]):
        count = lcs(i - 1, j - 1, count + 1)
 
    count = max(count, max(lcs(i, j - 1, 0),
                           lcs(i - 1, j, 0)))
 
    return count
def LCSubStr(X, Y, m, n): 
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains the
    # length of longest common suffix of
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program. 
    # LCSuff is the table with zero
    # value initially in each cell
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
 
    # To store the length of
    # longest common substring
    result = 0
 
    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result

# Python3 program to find Minimum 
# number of jumps to reach end 
# Returns minimum number of jumps
# to reach arr[n-1] from arr[0]
def minJumps(arr, n):
    jumps = [0 for i in range(n)] 
    if (n == 0) or (arr[0] == 0):
        return float('inf') 
    jumps[0] = 0 
    # Find the minimum number of 
    # jumps to reach arr[i] from 
    # arr[0] and assign this 
    # value to jumps[i]
    for i in range(1, n):
        jumps[i] = float('inf')
        for j in range(i):
            if (i <= j + arr[j]) and (jumps[j] != float('inf')):
                jumps[i] = min(jumps[i], jumps[j] + 1)
                break
    return jumps[n-1]

# Python3 code for above approach  
def idToShortURL(id): 
    map = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    shortURL = "" 
      
    # for each digit find the base 62 
    while(id > 0): 
        shortURL += map[id % 62] 
        id //= 62
  
    # reversing the shortURL 
    return shortURL[len(shortURL): : -1] 
  
def shortURLToId(shortURL): 
    id = 0
    for i in shortURL: 
        val_i = ord(i) 
        if(val_i >= ord('a') and val_i <= ord('z')): 
            id = id*62 + val_i - ord('a') 
        elif(val_i >= ord('A') and val_i <= ord('Z')): 
            id = id*62 + val_i - ord('Z') + 26
        else: 
            id = id*62 + val_i - ord('0') + 52
    return id

# A O(n ^ 2) time and O(1) space program to find the  
# longest palindromic substring   
# This function prints the longest palindrome substring (LPS) 
# of str[]. It also returns the length of the longest palindrome 
def longestPalSubstr(string): 
    maxLength = 1  
    start = 0
    length = len(string)   
    low = 0
    high = 0  
    # One by one consider every character as center point of  
    # even and length palindromes 
    for i in xrange(1, length): 
        # Find the longest even length palindrome with center 
    # points as i-1 and i. 
        low = i - 1
        high = i 
        while low >= 0 and high < length and string[low] == string[high]: 
            if high - low + 1 > maxLength: 
                start = low 
                maxLength = high - low + 1
            low -= 1
            high += 1  
        # Find the longest odd length palindrome with center  
        # point as i 
        low = i - 1
        high = i + 1
        while low >= 0 and high < length and string[low] == string[high]: 
            if high - low + 1 > maxLength: 
                start = low 
                maxLength = high - low + 1
            low -= 1
            high += 1  
    print "Longest palindrome substring is:", 
    print string[start:start + maxLength]   
    return maxLength 

# Python program to remove 
# all adjacent duplicates from a string 
# Recursively removes adjacent 
# duplicates from str and returns
# new string. las_removed is a 
# pointer to last_removed character
def removeUtil(string, last_removed): 
    # If length of string is 1 or 0
    if len(string) == 0 or len(string) == 1:
        return string
 
    # Remove leftmost same characters 
    # and recur for remaining 
    # string
    if string[0] == string[1]:
        last_removed = ord(string[0])
        while len(string) > 1 and string[0] ==
                                    string[1]:
            string = string[1:]
        string = string[1:]
 
        return removeUtil(string, last_removed) 
    # At this point, the first 
    # character is definiotely different
    # from its adjacent. Ignore first 
    # character and recursively 
    # remove characters from remaining string
    rem_str = removeUtil(string[1:], last_removed) 
    # Check if the first character 
    # of the rem_string matches 
    # with the first character of 
    # the original string
    if len(rem_str) != 0 and rem_str[0] ==
                                         string[0]:
        last_removed = ord(string[0])
        return (rem_str[1:])
 
    # If remaining string becomes 
    # empty and last removed character
    # is same as first character of 
    # original string. This is needed
    # for a string like "acbbcddc"
    if len(rem_str) == 0 and last_removed ==
                                   ord(string[0]):        return rem_str 
    # If the two first characters of 
    # str and rem_str don't match, 
    # append first character of str 
    # before the first character of 
    # rem_str.
    return ([string[0]] + rem_str) 
def remove(string):
    last_removed = 0
    return toString(removeUtil(toList(string), 
                                    last_removed))
 
# Utility functions
def toList(string):
    x = []
    for i in string:
        x.append(i)
    return x
 
def toString(x):
    return ''.join(x)
 
# Python3 implementation of above approach 
# Method to remove duplicates 
def removeDuplicatesFromString(string):     
    # Table to keep track of visited 
    # characters
    table = [0 for i in range(256)]     
    # To keep track of end index 
    # of resultant string
    endIndex = 0
    string = list(string)     
    for i in range(len(string)):
        if (table[ord(string[i])] == 0):
            table[ord(string[i])] = -1
            string[endIndex] = string[i]
            endIndex += 1             
    ans = ""
    for i in range(endIndex):
      ans += string[i]       
    return ans

def binary_search(list, item):
    low = 0
    high = len(list)—1
    while low <= high:
        mid = (low + high)
        guess = list[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None
my_list = [1, 3, 5, 7, 9]
print binary_search(my_list, 3) # => 1
print binary_search(my_list, -1) # => None
def selectionSort(arr): Sorts an array
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))
    return newArr
def findSmallest(arr):
    smallest = arr[0] Stores the smallest value
    smallest_index = 0 Stores the index of the smallest value
    for i in range(1, len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index
print selectionSort([5, 3, 6, 2, 10]
// breath first search-shortest path,graph contains only segment = {} graph[“you”] = [“alice”, “bob”, “claire”]
def search(name):
    search_queue = deque()
    search_queue += graph[name]
    searched = []
    while search_queue:
        person = search_queue.popleft()
        if not person in searched:
            if person_is_seller(person):
                print person + “ is a mango seller!”
                return True
            else:
                search_queue += graph[person]
                searched.append(person)
    return False
//shortest path including cost-graph[“start”] = {},graph[“start”][“a”] = 6(neighbour hash),
costs = {},costs[“a”] = 6,parents = {},parents[“a”] = “start”
def shortestmincostPath():
    node = find_lowest_cost_node(costs)
    while node is not None:
        cost = costs[node]
        neighbors = graph[node]
        for n in neighbors.keys():
            new_cost = cost + neighbors[n]
            if costs[n] > new_cost:
                costs[n] = new_cost
                parents[n] = node
        processed.append(node)
        node = find_lowest_cost_node(costs)
    return processed
def find_lowest_cost_node(costs):
    lowest_cost = float(“inf”)
    lowest_cost_node = None
    for node in costs: Go through each node.
        cost = costs[node]
        if cost < lowest_cost and node not in processed:
            lowest_cost = cost … set it as the new lowest-cost node.
            lowest_cost_node = node
    return lowest_cost_node
//greedy-min stations to cover all cities-stations = {},stations[“kone”] = set([“id”, “nv”, “ut”]),stations[“ktwo”] = set([“wa”, “id”, “mt”])
def minStation(stations):
        while states_needed:
            best_station = None
            states_covered = set()
            for station, states in stations.items():
                covered = states_needed & states
                if len(covered) > len(states_covered):
                    best_station = station
                    states_covered = covered
        states_needed -= states_covered
        final_stations.add(best_station)
        return final_stations
//dynamic programming-longest common substring
if word_a[i] == word_b[j]:
    cell[i][j] = cell[i-1][j-1] + 1
else:
    cell[i][j] = 0
//dynamic programming-longest common subsequence
cell = [[0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word_a[i] == word_b[j]:
                cell[i][j] = cell[i-1][j-1] + 1
            else:
                cell[i][j] = max(cell[i-1][j], cell[i][j-1])
return cell[n][m]
//lis
lis = [1]*n,maximum=0
for i in range (1 , n):
    for j in range(0 , i):
        if arr[i] > arr[j] and lis[i]< lis[j] + 1 :
            lis[i] = lis[j]+1
for i in range(n):
    maximum = max(maximum , lis[i])
return maximum
//remove duplicates and sorted
s = input()
words = [word for word in s.split(" ")]
print(" ".join(sorted(list(set(words)))))
//square each odd number in lists
numbers = [x for x in values.split(",") if int(x)%2!=0]
print(",".join(numbers)
//sort tuple-name,age,score
l = []
while True:
    s = input()
    if not s:
        break
    l.append(tuple(s.split(",")))
print(sorted(l, key=itemgetter(0,1,2)))
//frequence of words in text
freq = {}   # frequency of words in text
line = input()
for word in line.split():
    freq[word] = freq.get(word,0)+1
words = freq.keys()
words.sort()
// filter list even numbers
li = [1,2,3,4,5,6,7,8,9,10]
evenNumbers = filter(lambda x: x%2==0, li)
//generat square of list via map function
squaredNumbers = map(lambda x: x**2, li)
//use map & filter to generate square of even numbers
evenNumbers = map(lambda x: x**2, filter(lambda x: x%2==0, li))
---squaredNumbers = map(lambda x: x**2, range(1,21))
---emailAddress = raw_input()
    pat2 = "(\w+)@((\w+\.)+(com))"
    r2 = re.match(pat2,emailAddress)
//fibonacci
def f(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return f(n-1)+f(n-2)
n=int(input())
values = [str(f(x)) for x in range(0, n+1)]
//remove 0.4.5th items
li = [x for (i,x) in enumerate(li) if i not in (0,4,5)]
----print(list(itertools.permutations([1,2,3])))
def solve(numheads,numlegs):
    for i in range(numheads+1):
        j=numheads-i
        if 2*i+4*j==numlegs:
            return i,j

-----coin change
dp = [[0 for x in range(len)] for x in range(amount+1)]
for i in range(len):
    dp[0][i] = 1
for i in range(1, amount+1):
    for j in range(len):
        x = dp[i - Coin[j]][j] if i-Coin[j] >= 0 else 0
        y = dp[i][j-1] if j >= 1 else 0
        dp[i][j] = x + y
return table[n][m-1]
def maxCapacity(W, wt, val, n=len(val)):
    dp = [[0 for x in range(W+1)] for x in range(n+1)]
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
    return K[n][W]
def cutRod(price, n):
    val = [0 for x in range(n+1)]
    for i in range(1, n+1):
        max_val = INT_MIN
        for j in range(i):
             max_val = max(max_val, price[j] + val[i-j-1])
        val[i] = max_val
    return val[n]
def staircaseways(steps, height):
    dp = [0 for i in range(height)]
    for s in steps:
        if s <= height:
            dp[s - 1] = 1
    for i in range(height):
        for s in steps:
            if i - s >= 0:
                dp[i] += dp[i - s]
    return dp[height - 1]
def minjumps(nums):
    dp = [-1]*n, dp[0] = 0
    for i in range(n):
        this_jump = i + nums[i]
        jumps = dp[i] + 1
        if this_jump >= n - 1:
            return jumps
        for j in range(this_jump, i, -1):
            if dp[j] != -1:
                break
            dp[j] = jumps
def max_subarray_sum(a):
    curr_sum = 0  max_sum = 0
    for val in a:
        # extend the current sum with the curren value;
        # reset it to 0 if it is smaller than 0, we care only about non-negative sums
        curr_sum = max(0, curr_sum + val)

        # check if this is the max sum
        max_sum = max(max_sum, curr_sum)

    return max_sum
def split_coinspartitionmindifference(coins):
    if len(coins) == 0:
        return -1
    full_sum = sum(coins)
    half_sum = full_sum // 2 + 1

    dp = [False]*half_sum
    dp[0] = True

    for c in coins:
        for i in range(half_sum - 1, -1, -1):
            if (i >= c) and dp[i - c]:
                # if you want to find coins, save the coin here dp[i] = c
                dp[i] = True

    for i in range(half_sum - 1, -1, -1):
        if dp[i]:
            # if you want to print coins, while i>0: print(dp[i]) i -= dp[i]
            return full_sum - 2 * i
def max_profitstock(prices):
    total = 0

    for i in range(1, len(prices)):
        total += max(0, prices[i] - prices[i - 1])

    return total
def max_area_containermostwater(height):
    l = 0
    r = len(height) - 1
    max_height = 0

    while l < r:
        left = height[l]
        right = height[r]

        current_height = min(left, right) * (r - l)
        max_height = max(max_height, current_height)

        # take the smaller side and search for a bigger height on that side
        if left < right:
            while (l < r) and (left >= height[l]):
                l += 1
        else:
            while (l < r) and (right >= height[r]):
                r -= 1

    return max_height
def count_tripletssumk(arr, k):
    count = 0
    n = len(arr)

    for i in range(n - 2):
        elements = set()
        curr_sum = k - arr[i]

        for j in range(i + 1, n):
            if (curr_sum - arr[j]) in elements:
                count += 1
            elements.add(arr[j])

    return count
def missing_number(nums):
    s = sum(nums)
    n = len(nums) + 1
    # sum formula (sum of the first n numbers) = (N*(N+1))/2
    return n * (n + 1) // 2 - s
def find_peak_elementindex(nums):
    l = 0
    r = len(nums) - 1

    while l < r:
        mid = (l + r) // 2
        if nums[mid] > nums[mid + 1]:
            # go left if the current value is smaller than the next one
            # in this moment you're sure that there is a peak element left from this one
            r = mid
        else:
            # go right if the current value is smaller than the next one
            # if the l comes to the end and all elements were in ascending order, then the last one is peak (because nums[n] is negative infinity)
            l = mid + 1

    return l
def find_kth_smallestnumber(arr, k):
    n = len(arr)
    if k > n:
        return None
    if k < 1:
        return None
    return kth_smallest(arr, k - 1, 0, n - 1)
def longest_increasing_subarraylen(arr):
    n = len(arr)
    longest = 0
    current = 1
    i = 1

    while i < n:
        if arr[i] < arr[i - 1]:
            longest = max(longest, current)
            current = 1
        else:
            current += 1

        i += 1

    # check again for max, maybe the last element is a part of the longest subarray
    return max(longest, current)
def majority_element_1(nums):
    nums.sort()
    return nums[len(nums) // 2]
def min_swapstosort(a):
    n = len(a)
    swaps = 0

    for i in range(n):
        # swap the elements till the right element isn't found
        while a[i] - 1 != i:
            swap = a[i] - 1
            # swap the elements
            a[swap], a[i] = a[i], a[swap]

            swaps += 1

    return swaps
def reverse_arr(arr):
    start = 0
    end = len(arr) - 1

    while start < end:
        # reverse the array from the start index to the end index by
        # swaping each element with the pair from the other part of the array
        swap(arr, start, end)
        start += 1
        end -= 1

    return arr
def rotate_array_1(arr, k, right = True):
    n = len(arr)
    right %= n

    # going right for K places is same like going left for N-K places
    if right:
        k = n - k

    # the shortest way to swap 2 parts of the array
    return arr[k:] + arr[:k]
def find_subarrayindexes(arr, k):
    n = len(arr)

    if n == 0:
        return -1

    start = 0
    end = 0
    current_sum = arr[0]

    while end < n:
        if current_sum == k:
            return (start + 1, end + 1)

        if current_sum < k:
            end += 1
            current_sum += arr[end]
        else:
            current_sum -= arr[start]
            start += 1
def bfs(graph,start):
    queue = [start]
    visited = []

    while queue:
        a = queue.pop(0)
        if a not in visited:
            visited.append(a)
            for neighbor in graph[a]:
                queue.append(neighbor)
    print visited
def isCyclic(self):
        # Mark all the vertices as not visited
        visited =[False]*(self.V)
        # Call the recursive helper function to detect cycle in different
        #DFS trees
        for i in range(self.V):
            if visited[i] ==False: #Don't recur for u if it is already visited
                if(self.isCyclicUtil(i,visited,-1))== True:
                    return True

    return False
def DFS(self,v):

    # Mark all the vertices as not visited
    visited = [False]*(len(self.graph))
    self.DFSUtil(v,visited)
def DFSUtil(self,v,visited):

    # Mark the current node as visited and print it
    visited[v]= True
    print v,

    # Recur for all the vertices adjacent to this vertex
    for i in self.graph[v]:
        if visited[i] == False:
            self.DFSUtil(i, visited)
    def topologicalSortUtil(self,v,visited,stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)

        # Push current vertex to stack which stores result
        stack.insert(0,v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False]*self.V
        stack =[]

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] == False:
                self.topologicalSortUtil(i,visited,stack)
    def middle(self):
        fast = self.head
        slow = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        print slow.data
    def reverse(self,cur):
        cur = cur
        n = cur.next

        if n.next:
            self.reverse(cur.next)
            n.next = cur
            cur.next = None
        else:
            n.next = cur
            self.head = n
def reverse(s):
    return s[::-1]
def isPalindrome(s):
    # Convert s to uppercase to ignore case sensitivity
    s = s.upper()
    # Checking if both string are equal or not
    if (s == reverse(s)):
        return True
    return False
String
can_string_be_rearranged_as_palindrome_counter
return sum(c % 2 for c in Counter(input_str.replace(" ", "").lower()).values()) < 2
capitalize the first letter of a sentence or a word
lower_to_upper = {lc: uc for lc, uc in zip(ascii_lowercase, ascii_uppercase)}
return lower_to_upper.get(sentence[0], sentence[0]) + sentence[1:]
check_anagrams
return (
        "".join(sorted(first_str.lower())).strip()
        == "".join(sorted(second_str.lower())).strip()
    )
check_pangram
flag = [False] * 26
    for char in input_str:
        if char.islower():
            flag[ord(char) - ord("a")] = True
    return all(flag)
is_palindrome
 s = "".join([character for character in s.lower() if character.isalnum()])
 return s == s[::-1]
string to lowecase letters
return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in word)
string to uppercase letters
return "".join(chr(ord(char) - 32) if "a" <= char <= "z" else char for char in word)
remove_duplicates
return " ".join(sorted(set(sentence.split())))
Reverses letters in a given string without adjusting the position of the words
return " ".join([word[::-1] for word in input_str.split()])
Reverses words in a given string
return " ".join(input_str.split()[::-1])
word_occurence
 occurrence = defaultdict(int)
    for word in sentence.split():
        occurrence[word] += 1
    return occurrence

def get_permutations(string):
    # Base case
    if len(string) <= 1:
        return set([string])

    all_chars_except_last = string[:-1]
    last_char = string[-1]

    # Recursive call: get all possible permutations for all chars except last
    permutations_of_all_chars_except_last = get_permutations(all_chars_except_last)

    # Put the last char in all possible positions for each of
    # the above permutations
    permutations = set()
    for permutation_of_all_chars_except_last in permutations_of_all_chars_except_last:
        for position in range(len(all_chars_except_last) + 1):
            permutation = (
                permutation_of_all_chars_except_last[:position]
                + last_char
                + permutation_of_all_chars_except_last[position:]
            )
            permutations.add(permutation)

    return permutations

Matrix
def count_islands(self) -> int:  # And finally, count all islands.
        visited = [[False for j in range(self.COL)] for i in range(self.ROW)]
        count = 0
        for i in range(self.ROW):
            for j in range(self.COL):
                if visited[i][j] is False and self.graph[i][j] == 1:
                    self.diffs(i, j, visited)
                    count += 1
        return count
def transpose(matrix: list[list], return_map: bool = True) -> list[list]:
    if _check_not_integer(matrix):
        if return_map:
            return map(list, zip(*matrix))
        else:
            return list(map(list, zip(*matrix)))
def transpose(matrix: [[]]) -> [[]]:
    matrix[:] = [list(x) for x in zip(*matrix)]
    return matrix
def reverse_row(matrix: [[]]) -> [[]]:
    matrix[:] = matrix[::-1]
    return matrix
def reverse_column(matrix: [[]]) -> [[]]:
    matrix[:] = [x[::-1] for x in matrix]
    return matrix
rotate_90 return reverse_row(transpose(matrix))
rotate_180 return reverse_row(reverse_column(matrix))
rotate_270  return reverse_column(transpose(matrix))
search_in_a_sorted_matrix
 i, j = m - 1, 0
    while i >= 0 and j < n:
        if key == mat[i][j]:
            print(f"Key {key} found at row- {i + 1} column- {j + 1}")
            return
        if key < mat[i][j]:
            i -= 1
        else:
            j += 1
List employees (names) who have a bigger salary than their boss

SELECT e.name AS 'Employee Name', e2.name AS 'Boss', 
e.salary AS 'Employee salary', e2.salary AS 'Boss salary'
FROM employees e
JOIN employees e2 ON e.boss_id = e2.employee_id
WHERE e2.salary < e.salary;

List employees who have the biggest salary IN their departments

Returns only one person for each department with the highest salary:

SELECT * 
FROM ( SELECT dept.Name Department, emp.Name Employee, emp.Salary Salary
       FROM Departments dept 
       JOIN Employees emp ON emp.department_id = dept.department_id
       ORDER BY salary desc ) result 
GROUP BY Department;

Returns one or more people for each department with the highest salary:

SELECT result.Name Department, emp2.Name Employee, result.salary Salary 
FROM ( SELECT dept.name, dept.department_id, max(emp1.salary) salary 
       FROM Departments dept 
       JOIN Employees emp1 ON emp1.department_id = dept.department_id 
       GROUP BY dept.name, dept.department_id ) result 
JOIN Employees emp2 ON emp2.department_id = result.department_id 
WHERE emp2.salary = result.salary;

List departments that have less than 3 people IN it

SELECT d.Name AS 'Department'
FROM departments d JOIN employees e 
ON e.department_id = d.Department_id
GROUP BY d.department_id
HAVING COUNT(e.employee_id) < 3

List ALL departments along WITH the NUMBER OF people there (tricky - people often do an "inner join" leaving OUT empty departments)

SELECT d.name, COUNT(e.department_id) AS num_employees
FROM departments d LEFT JOIN employees e ON d.id=e.department_id
GROUP BY d.id

List employees that don't have a boss in the same department

SELECT e.name 
FROM employees e JOIN employees b ON e.boss_id=b.id
WHERE e.department_id != b.department_id
OR b.boss_id is NULL

List all departments along with the total salary there

SELECT d.name AS 'Department', SUM(e.salary) AS 'Total Salary'
FROM departments d LEFT OUTER JOIN employees e 
ON d.department_id = e.department_id
GROUP BY d.department_id

/* 1. Write a query in SQL to display the first name, last name, department number, and department name for each employee. */

SELECT e.first_name,
       e.last_name,
       e.department_id,
       d.department_name
  FROM employees e
  INNER JOIN departments d
    ON e.department_id = d.department_id;


/* 2. Write a query in SQL to display the first and last name, department, city, and state province for each employee. */

SELECT e.first_name,
       e.last_name,
       d.department_name,
       l.city,
       l.state_province
  FROM employees e
  INNER JOIN departments d
    ON e.department_id = d.department_id
  INNER JOIN locations l
    ON d.location_id = l.location_id;


/* 3. Write a query in SQL to display the first name, last name, salary, and job grade for all employees. */

SELECT e.first_name,
       e.last_name,
       e.salary,
       j.grade_level
  FROM employees e
  INNER JOIN job_grades j
    ON e.salary BETWEEN j.lowest_sal AND j.highest_sal;


/* 4. Write a query in SQL to display the first name, last name, department number and department name, for all employees for departments 80 or 40. */

SELECT e.first_name,
       e.last_name,
       d.department_id,
       d.department_name
  FROM employees e
  INNER JOIN departments d
    ON e.department_id = d.department_id
      AND d.department_id IN (80, 40)
  ORDER BY e.last_name;


/* 5. Write a query in SQL to display those employees who contain a letter z to their first name and also display their last name, department, city, and state province. */

SELECT e.first_name,
       e.last_name,
       d.department_name,
       l.city,
       l.state_province
  FROM employees e
  INNER JOIN departments d
    ON e.department_id = d.department_id
  INNER JOIN locations l
    ON d.location_id = l.location_id
  WHERE e.first_name LIKE '%z%';


/* 6. Write a query in SQL to display all departments including those where does not have any employee. */

SELECT e.first_name,
       e.last_name,
       d.department_id,
       d.department_name
  FROM departments d
  LEFT JOIN employees e
    ON d.department_id = e.department_id;


/* 7. Write a query in SQL to display the first and last name and salary for those employees who earn less than the employee earn whose number is 182. */

SELECT e1.first_name,
       e1.last_name,
       e1.salary
  FROM employees e1
  INNER JOIN employees e2
    ON e1.salary < e2.salary
      AND e2.employee_id = 182;


/* 8. Write a query in SQL to display the first name of all employees including the first name of their manager. */

SELECT e1.first_name AS "employee_name",
       e2.first_name AS "manager_name"
  FROM employees e1
  INNER JOIN employees e2
    ON e1.manager_id = e2.employee_id;


/* 9. Write a query in SQL to display the department name, city, and state province for each department. */

SELECT d.department_name,
       l.city,
       l.state_province
  FROM departments d
  INNER JOIN locations l
    ON d.location_id = l.location_id;


/* 10. Write a query in SQL to display the first name, last name, department number and name, for all employees who have or have not any department. */

SELECT e.first_name,
       e.last_name,
       d.department_id,
       d.department_name
  FROM employees e
  LEFT JOIN departments d
    ON e.department_id = d.department_id;


/* 11. Write a query in SQL to display the first name of all employees and the first name of their manager including those who does not working under any manager. */

SELECT e1.first_name AS "employee_name",
       e2.first_name AS "manager_name"
  FROM employees e1
  LEFT JOIN employees e2
    ON e1.manager_id = e2.employee_id;


/* 12. Write a query in SQL to display the first name, last name, and department number for those employees who works in the same department as the employee who holds the last name as Taylor. */

SELECT e1.first_name,
       e1.last_name,
       e1.department_id
  FROM employees e1
  INNER JOIN employees e2
    ON e1.department_id = e2.department_id
      AND e2.last_name = 'Taylor';


/* 13. Write a query in SQL to display the job title, department name, full name (first and last name ) of employee, and starting date for all the jobs which started on or after 1st January, 1993 and ending with on or before 31 August, 1997 */

SELECT j.job_title,
       d.department_name,
       CONCAT(e.first_name, ' ', e.last_name) AS full_name,
       jh.start_date
  FROM employees e
  INNER JOIN job_history jh
    ON e.employee_id = jh.employee_id
      AND jh.start_date BETWEEN '1993-01-01' AND '1997-08-31'
  INNER JOIN jobs j
    ON jh.job_id = j.job_id
  INNER JOIN departments d
    ON jh.department_id = d.department_id;


/* 14. Write a query in SQL to display job title, full name (first and last name ) of employee, and the difference between maximum salary for the job and salary of the employee. */

SELECT j.job_title,
       CONCAT(e.first_name, ' ', e.last_name) AS full_name,
       (j.max_salary - e.salary) AS salary_diff
  FROM employees e
  INNER JOIN jobs j
    ON e.job_id = j.job_id;

-- Note: This also works using a NATURAL JOIN which creates an implicit join based on the common columns in the two tables being joined.

SELECT j.job_title,
       CONCAT(e.first_name, ' ', e.last_name) AS full_name,
       (j.max_salary - e.salary) AS salary_diff
  FROM employees e
  NATURAL INNER JOIN jobs j;


/* 15. Write a query in SQL to display the name of the department, average salary and number of employees working in that department who got commission. */

SELECT d.department_name,
       AVG(e.salary),
       COUNT(commission_pct)
  FROM employees e
  JOIN departments d
    ON e.department_id = d.department_id
  GROUP BY d.department_name;


/* 16. Write a query in SQL to display the full name (first and last name ) of employees, job title and the salary differences to their own job for those employees who is working in the department ID 80. */

SELECT CONCAT(e.first_name, ' ', e.last_name) AS full_name,
       j.job_title,
       (j.max_salary - e.salary) AS salary_diff
  FROM employees e
  INNER JOIN jobs j
    ON e.job_id = j.job_id
      WHERE e.department_id = 80;


/* 17. Write a query in SQL to display the name of the country, city, and the departments which are running there. */

SELECT c.country_name,
       l.city,
       d.department_name
  FROM countries c
  INNER JOIN locations l
    ON c.country_id = l.country_id
  INNER JOIN departments d
    ON l.location_id = d.location_id;

-- Note: This also works using JOIN and USING on the common columns.

SELECT c.country_name,
       l.city,
       d.department_name
  FROM countries c
  INNER JOIN locations l USING (country_id) 
  INNER JOIN departments d USING (location_id);


/* 18. Write a query in SQL to display department name and the full name (first and last name) of the manager. */

SELECT d.department_name,
       CONCAT(e.first_name, ' ', e.last_name) AS full_name
  FROM departments d
  INNER JOIN employees e
    ON d.manager_id = e.employee_id;


/* 19. Write a query in SQL to display job title and average salary of employees. */

SELECT j.job_title,
       AVG(e.salary)
  FROM employees e
  INNER JOIN jobs j
    ON e.job_id = j.job_id
  GROUP BY j.job_title;


/* 20. Write a query in SQL to display the details of jobs which was done by any of the employees who is presently earning a salary on and above 12000. */

SELECT jh.*
  FROM employees e
  INNER JOIN job_history jh
    ON e.employee_id = jh.employee_id
      WHERE salary >= 12000.00;


/* 21. Write a query in SQL to display the country name, city, and number of those departments where at least 2 employees are working. */

SELECT c.country_name,
       l.city,
       COUNT(d.department_id)
  FROM countries c
  INNER JOIN locations l
    ON c.country_id = l.country_id
  INNER JOIN departments d
     ON l.location_id = d.location_id
  WHERE d.department_id IN (SELECT e.department_id
                              FROM employees e
                              GROUP BY e.department_id 
                              HAVING COUNT(e.department_id) >= 2)
  GROUP BY c.country_name, l.city;


/* 22. Write a query in SQL to display the department name, full name (first and last name) of manager, and their city.  */

SELECT d.department_name,
       CONCAT(e.first_name, ' ', e.last_name) AS full_name,
       l.city
  FROM employees e
  INNER JOIN departments d
    ON e.employee_id = d.manager_id
  INNER JOIN locations l
    ON d.location_id = l.location_id;


/* 23. Write a query in SQL to display the employee ID, job name, number of days worked in for all those jobs in department 80. */

SELECT jh.employee_id,
       j.job_title,
       (jh.end_date - jh.start_date) AS num_days
  FROM jobs j
  INNER JOIN job_history jh
    ON j.job_id = jh.job_id
      WHERE jh.department_id = 80;

-- Note: This also works using a NATURAL JOIN which creates an implicit join based on the common columns in the two tables being joined.

SELECT jh.employee_id,
       j.job_title,
       (jh.end_date - jh.start_date) AS num_days
  FROM jobs j
  NATURAL INNER JOIN job_history jh
  WHERE jh.department_id = 80;


/* 24. Write a query in SQL to display the full name (first and last name), and salary of those employees who working in any department located in London. */

SELECT CONCAT(e.first_name, ' ', e.last_name) AS full_name,
       e.salary
  FROM employees e
  INNER JOIN departments d
    ON e.department_id = d.department_id
  INNER JOIN locations l
    ON d.location_id = l.location_id
      WHERE l.city = 'London';


/* 25. Write a query in SQL to display full name(first and last name), job title, starting and ending date of last jobs for those employees with worked without a commission percentage. */

SELECT CONCAT(e.first_name, ' ', e.last_name) AS full_name,
       j.job_title,
       jh.*
  FROM employees e
  INNER JOIN (SELECT MAX(start_date) AS starting_date,
                     MAX(end_date) AS ending_date,
                     employee_id
                FROM job_history
                GROUP BY employee_id) jh
    ON e.employee_id = jh.employee_id
  INNER JOIN jobs j
    ON e.job_id = j.job_id
      WHERE e.commission_pct = 0;


/* 26. Write a query in SQL to display the department name and number of employees in each of the department. */

SELECT d.department_name,
       COUNT(e.employee_id) AS num_employees
  FROM departments d
  INNER JOIN employees e
    ON d.department_id = e.department_id
  GROUP BY d.department_name;


/* 27. Write a query in SQL to display the full name (first and last name) of employee with ID and name of the country presently where (s)he is working. */

SELECT CONCAT(e.first_name, ' ', e.last_name) AS full_name,
       e.employee_id,
       c.country_name
  FROM employees e
  INNER JOIN departments d
    ON e.department_id = d.department_id
  INNER JOIN locations l
    ON d.location_id = l.location_id
  INNER JOIN countries c
    ON l.country_id = c.country_id;

* 1. Write a query to display the name (first name and last name) for those employees who gets more salary than the employee whose ID is 163. */

SELECT first_name,
       last_name
  FROM employees
  WHERE salary > (SELECT salary
                    FROM employees
                    WHERE employee_id = 163);


/* 2. Write a query to display the name (first name and last name), salary, department id, job id for those employees who works in the same designation as the employee works whose id is 169. */

SELECT first_name,
       last_name,
       salary,
       department_id,
       job_id
  FROM employees
  WHERE job_id = (SELECT job_id
                    FROM employees
                    WHERE employee_id = 169);


/* 3. Write a query to display the name (first name and last name), salary, department id for those employees who earn such amount of salary which is the smallest salary of any of the departments. */

SELECT first_name,
       last_name,
       salary,
       department_id
  FROM employees
  WHERE salary IN (SELECT MIN(salary)
                     FROM employees
                     GROUP BY department_id);


/* 4. Write a query to display the employee id, employee name (first name and last name) for all employees who earn more than the average salary. */

SELECT employee_id,
       first_name,
       last_name
  FROM employees
  WHERE salary > (SELECT AVG(salary)
                    FROM employees);


/* 5. Write a query to display the employee name (first name and last name), employee id and salary of all employees who report to Payam. */

SELECT first_name,
       last_name,
       employee_id,
       salary
  FROM employees
  WHERE manager_id = (SELECT employee_id
                        FROM employees
                        WHERE first_name = 'Payam');


/* 6. Write a query to display the department number, name (first name and last name), job_id and department name for all employees in the Finance department. */

SELECT e.department_id,
       e.first_name,
       e.last_name,
       e.job_id,
       d.department_name
  FROM employees e
  INNER JOIN departments d
    ON e.department_id = d.department_id
  WHERE d.department_name = 'Finance';


/* 7. Write a query to display all the information of an employee whose salary and reporting person id is 3000 and 121, respectively. */

SELECT *
  FROM employees
  WHERE salary = 3000.00
    AND manager_id = 121;

-- Note: This also works using subquery.

SELECT *
  FROM employees 
  WHERE (salary, manager_id) = (SELECT 3000, 121);


/* 8. Display all the information of an employee whose id is any of the number 134, 159 and 183. */

SELECT *
  FROM employees
  WHERE employee_id IN (134, 159, 183);


/* 9. Write a query to display all the information of the employees whose salary is within the range 1000 and 3000. */

SELECT *
  FROM employees
  WHERE salary BETWEEN 1000.00 AND 3000.00;


/* 10. Write a query to display all the information of the employees whose salary is within the range of smallest salary and 2500. */

SELECT *
  FROM employees
  WHERE salary BETWEEN (SELECT MIN(salary)
                          FROM employees) AND 2500.00;


/* 11. Write a query to display all the information of the employees who does not work in those departments where some employees works whose manager id within the range 100 and 200. */

SELECT *
  FROM employees
  WHERE department_id NOT IN (SELECT department_id
                                FROM departments
                                WHERE manager_id BETWEEN 100 AND 200);


/* 12. Write a query to display all the information for those employees whose id is any id who earn the second highest salary. */

SELECT *
  FROM employees
  WHERE employee_id IN (SELECT employee_id
                          FROM employees
                          WHERE salary IN (SELECT MAX(salary)
                                             FROM employees
                                             WHERE salary < (SELECT MAX(salary)
                                                               FROM employees)));


/* 13. Write a query to display the employee name (first name and last name) and hire date for all employees in the same department as Clara. Exclude Clara. */

SELECT first_name,
       last_name,
       hire_date
  FROM employees
  WHERE department_id = (SELECT department_id
                           FROM employees
                           WHERE first_name = 'Clara')
    AND first_name != 'Clara';


/* 14. Write a query to display the employee number and name (first name and last name) for all employees who work in a department with any employee whose name contains a T. */

SELECT employee_id,
       first_name,
       last_name
  FROM employees
  WHERE department_id IN (SELECT department_id
                            FROM employees
                            WHERE first_name LIKE '%T%');


/* 15. Write a query to display the employee number, name (first name and last name), and salary for all employees who earn more than the average salary and who work in a department with any employee with a J in their name. */

SELECT employee_id,
       first_name,
       last_name,
       salary
  FROM employees
  WHERE salary > (SELECT AVG(salary)
                    FROM employees)
    AND department_id IN (SELECT department_id
                            FROM employees
                            WHERE first_name LIKE '%J%');


/* 16. Display the employee name (first name and last name), employee id, and job title for all employees whose department location is Toronto. */

SELECT first_name,
       last_name,
       employee_id,
       job_id
  FROM employees
  WHERE department_id IN (SELECT department_id
                            FROM departments
                            WHERE location_id IN (SELECT location_id
                                                    FROM locations
                                                    WHERE city = 'Toronto'));


/* 17. Write a query to display the employee number, name (first name and last name) and job title for all employees whose salary is smaller than any salary of those employees whose job title is MK_MAN. */

SELECT employee_id,
       first_name,
       last_name,
       job_id
  FROM employees
  WHERE salary < ANY (SELECT salary
                        FROM employees
                        WHERE job_id = 'MK_MAN');


/* 18. Write a query to display the employee number, name (first name and last name) and job title for all employees whose salary is smaller than any salary of those employees whose job title is MK_MAN. Exclude Job title MK_MAN. */

SELECT employee_id,
       first_name,
       last_name,
       job_id
  FROM employees
  WHERE salary < ANY (SELECT salary
                        FROM employees
                        WHERE job_id = 'MK_MAN')
    AND job_id != 'MK_MAN';


/* 19. Write a query to display the employee number, name (first name and last name) and job title for all employees whose salary is more than any salary of those employees whose job title is PU_MAN. Exclude job title PU_MAN. */

SELECT employee_id,
       first_name,
       last_name,
       job_id
  FROM employees
  WHERE salary > ANY (SELECT salary
                        FROM employees
                        WHERE job_id = 'PU_MAN')
    AND job_id != 'PU_MAN';


/* 20. Write a query to display the employee number, name (first name and last name) and job title for all employees whose salary is more than any average salary of any department. */

SELECT employee_id,
       first_name,
       last_name,
       job_id
  FROM employees
  WHERE salary > ANY (SELECT AVG(salary)
                        FROM employees
                        GROUP BY department_id);


/* 21. Write a query to display the employee name( first name and last name ) and department for all employees for any existence of those employees whose salary is more than 3700. */

SELECT first_name,
       last_name,
       department_id
  FROM employees
  WHERE EXISTS (SELECT *
                  FROM employees
                  WHERE salary > 3700.00);


/* 22. Write a query to display the department id and the total salary for those departments which contains at least one employee. */

SELECT department_id,
       SUM(salary)
  FROM employees
  WHERE department_id IN (SELECT department_id
                            FROM departments)
  GROUP BY department_id
  HAVING COUNT(department_id) >= 1;


/* 23. Write a query to display the employee id, name (first name and last name) and the job id column with a modified title SALESMAN for those employees whose job title is ST_MAN and DEVELOPER for whose job title is IT_PROG. */

SELECT employee_id,
       first_name,
       last_name,
       CASE WHEN job_id = 'ST_MAN' THEN 'SALESMAN'
            WHEN job_id = 'IT_PROG' THEN 'DEVELOPER'
            ELSE job_id END AS job_id_mod
  FROM employees;


/* 24. Write a query to display the employee id, name (first name and last name), salary and the SalaryStatus column with a title HIGH and LOW respectively for those employees whose salary is more than and less than the average salary of all employees. */

SELECT employee_id,
       first_name,
       last_name,
       salary,
       CASE WHEN salary >= (SELECT AVG(salary) FROM employees) THEN 'HIGH'
            ELSE 'LOW' END AS salary_status
  FROM employees;


/* 25. Write a query to display the employee id, name (first name and last name), Salary, AvgCompare (salary - the average salary of all employees) and the SalaryStatus column with a title HIGH and LOW respectively for those employees whose salary is more than and less than the average salary of all employees. */

SELECT employee_id,
       first_name,
       last_name,
       salary AS salary_drawn,
       ROUND(salary - (SELECT AVG(salary) FROM employees), 2) AS avg_compare,
       CASE WHEN salary >= (SELECT AVG(salary) FROM employees) THEN 'HIGH'
                 ELSE 'LOW' END AS salary_status
  FROM employees;


/* 26. Write a subquery that returns a set of rows to find all departments that do actually have one or more employees assigned to them. */

SELECT department_name
  FROM departments
  WHERE department_id IN (SELECT DISTINCT department_id
                            FROM employees);


/* 27. Write a query that will identify all employees who work in departments located in the United Kingdom. */

SELECT *
  FROM employees
  WHERE department_id IN (SELECT department_id
                            FROM departments
                            WHERE location_id IN (SELECT location_id
                                                    FROM locations
                                                    WHERE country_id IN (SELECT country_id
                                                                           FROM countries
                                                                           WHERE country_name = 'United Kingdom')));


/* 28. Write a query to identify all the employees who earn more than the average and who work in any of the IT departments. */

SELECT *
  FROM employees
  WHERE salary > (SELECT AVG(salary)
                    FROM employees)
    AND department_id IN (SELECT department_id
                            FROM departments
                            WHERE department_name LIKE ('%IT%'));


/* 29. Write a query to determine who earns more than Mr. Ozer. */

SELECT first_name,
       last_name,
       salary
  FROM employees
  WHERE salary > (SELECT salary
                    FROM employees
                    WHERE last_name = 'Ozer');


/* 30. Write a query to find out which employees have a manager who works for a department based in the US. */

SELECT first_name,
       last_name
  FROM employees
  WHERE manager_id IN (SELECT employee_id
                         FROM employees
                         WHERE department_id IN (SELECT department_id
                                                   FROM departments
                                                   WHERE location_id IN (SELECT location_id
                                                                           FROM locations
                                                                           WHERE country_id = 'US')));


/* 31. Write a query which is looking for the names of all employees whose salary is greater than 50% of their department’s total salary bill. */

SELECT e1.first_name,
       e1.last_name
  FROM employees e1
  WHERE salary > (SELECT SUM(salary)*0.5
                    FROM employees e2
                    WHERE e1.department_id =  e2.department_id);


/* 32. Write a query to get the details of employees who are managers. */

SELECT *
  FROM employees
  WHERE employee_id IN (SELECT manager_id
                          FROM departments);


/* 33. Write a query to get the details of employees who manage a department. */

SELECT * 
  FROM employees 
  WHERE employee_id = ANY (SELECT manager_id
                             FROM departments);


/* 34. Write a query to display the employee id, name (first name and last name), salary, department name and city for all the employees who gets the salary as the salary earn by the employee which is maximum within the joining person January 1st, 2002 and December 31st, 2003. */

SELECT e.employee_id,
       e.first_name,
       e.last_name,
       e.salary,
       d.department_name,
       l.city
  FROM employees e
  INNER JOIN departments d
    ON e.department_id = d.department_id
  INNER JOIN locations l
    ON d.location_id = l.location_id
  WHERE e.salary = (SELECT MAX(salary)
                      FROM employees
                      WHERE hire_date BETWEEN '2002-01-01' AND '2003-12-31');


/* 35. Write a query in SQL to display the department code and name for all departments which located in the city London. */

SELECT department_id,
       department_name
  FROM departments
  WHERE location_id IN (SELECT location_id
                          FROM locations
                          WHERE city = 'London');


/* 36. Write a query in SQL to display the first and last name, salary, and department ID for all those employees who earn more than the average salary and arrange the list in descending order on salary. */

SELECT first_name,
       last_name,
       salary,
       department_id
  FROM employees
  WHERE salary > (SELECT AVG(salary)
        FROM employees)
  ORDER BY salary DESC;


/* 37. Write a query in SQL to display the first and last name, salary, and department ID for those employees who earn more than the maximum salary of a department which ID is 40. */

SELECT first_name,
       last_name,
       salary,
       department_id
  FROM employees
  WHERE salary > (SELECT MAX(salary)
        FROM employees
                    WHERE department_id = 40);


/* 38. Write a query in SQL to display the department name and Id for all departments where they located, that Id is equal to the Id for the location where department number 30 is located. */

SELECT department_name,
       department_id
  FROM departments
  WHERE location_id = (SELECT location_id
                         FROM departments 
                         WHERE department_id = 30);


/* 39. Write a query in SQL to display the first and last name, salary, and department ID for all those employees who work in that department where the employee works who hold the ID 201. */

SELECT first_name,
       last_name,
       salary,
       department_id
  FROM employees
  WHERE department_id = (SELECT department_id
               FROM employees
                           WHERE employee_id = 201);


/* 40. Write a query in SQL to display the first and last name, salary, and department ID for those employees whose salary is equal to the salary of the employee who works in that department which ID is 40. */

SELECT first_name,
       last_name,
       salary,
       department_id
  FROM employees
  WHERE salary IN (SELECT salary
         FROM employees
                     WHERE department_id = 40);


/* 41. Write a query in SQL to display the first and last name, and department code for all employees who work in the department Marketing. */

SELECT first_name,
       last_name,
       department_id
  FROM employees
  WHERE department_id IN (SELECT department_id
                FROM departments
                            WHERE department_name = 'Marketing');


/* 42. Write a query in SQL to display the first and last name, salary, and department ID for those employees who earn more than the minimum salary of a department which ID is 40. */

SELECT first_name,
       last_name,
       salary,
       department_id
  FROM employees
  WHERE salary > (SELECT MIN(salary)
        FROM employees
                WHERE department_id = 40);


/* 43. Write a query in SQL to display the full name, email, and hire date for all those employees who was hired after the employee whose ID is 165. */

SELECT CONCAT(first_name, ' ', last_name) AS full_name,
       email,
       hire_date
  FROM employees
  WHERE hire_date > (SELECT hire_date
                       FROM employees
                       WHERE employee_id = 165);


/* 44. Write a query in SQL to display the first and last name, salary, and department ID for those employees who earn less than the minimum salary of a department which ID is 70. */

SELECT first_name,
       last_name,
       salary,
       department_id
  FROM employees
  WHERE salary < (SELECT MIN(salary)
        FROM employees
                WHERE department_id = 70);


/* 45. Write a query in SQL to display the first and last name, salary, and department ID for those employees who earn less than the average salary, and also work at the department where the employee Laura is working as a first name holder. */

SELECT first_name,
       last_name,
       salary,
       department_id
  FROM employees
  WHERE salary < (SELECT AVG(salary)
        FROM employees)
    AND department_id = (SELECT department_id
                           FROM employees
                           WHERE first_name = 'Laura');


/* 46. Write a query in SQL to display the first and last name, salary, and department ID for those employees whose department is located in the city London. */

SELECT first_name,
       last_name,
       salary,
       department_id 
  FROM employees
  WHERE department_id IN (SELECT department_id
                            FROM departments
                            WHERE location_id IN (SELECT location_id
                                                    FROM locations
                                                    WHERE city = 'London'));


/* 47. Write a query in SQL to display the city of the employee whose ID 134 and works there. */

SELECT city
  FROM locations
  WHERE location_id IN (SELECT location_id
                          FROM departments
                          WHERE department_id IN (SELECT department_id
                                                    FROM employees
                                                    WHERE employee_id = 134));


/* 48. Write a query in SQL to display the the details of those departments which max salary is 7000 or above for those employees who already done one or more jobs. */

SELECT *
  FROM departments
  WHERE department_id IN (SELECT department_id
                            FROM employees
                            WHERE employee_id IN (SELECT employee_id
                                                    FROM job_history
                                                    GROUP BY employee_id
                                                    HAVING COUNT(*) > 1)
                            GROUP BY department_id
                            HAVING MAX(salary) > 7000);


/* 49. Write a query in SQL to display the detail information of those departments which starting salary is at least 8000. */

SELECT *
  FROM departments
  WHERE department_id IN (SELECT department_id
                            FROM employees
                            GROUP BY department_id
                            HAVING MIN(salary) >= 8000);


/* 50. Write a query in SQL to display the full name (first and last name) of manager who is supervising 4 or more employees. */

SELECT CONCAT(first_name, ' ', last_name) AS full_name
  FROM employees
  WHERE employee_id IN (SELECT manager_id
                          FROM employees
                          GROUP BY manager_id
                          HAVING COUNT(*) >= 4);


/* 51. Write a query in SQL to display the details of the current job for those employees who worked as a Sales Representative in the past. */

SELECT *
  FROM jobs
  WHERE job_id IN (SELECT job_id
                     FROM employees
                     WHERE employee_id IN (SELECT employee_id
                                             FROM job_history
                                             WHERE job_id = 'SA_REP'));


/* 52. Write a query in SQL to display all the information about those employees who earn second lowest salary of all the employees. */

SELECT *
  FROM employees
  WHERE salary IN (SELECT MIN(salary)
                     FROM employees
                     WHERE salary > (SELECT MIN(salary)
                                       FROM employees));


/* 53. Write a query in SQL to display the details of departments managed by Susan. */

SELECT *
  FROM departments
  WHERE manager_id IN (SELECT employee_id
                         FROM employees
                         WHERE first_name = 'Susan');


/* 54. Write a query in SQL to display the department ID, full name (first and last name), salary for those employees who is highest salary drawer in a department. */

SELECT department_id,
       CONCAT(first_name, ' ', last_name) AS full_name,
       salary
  FROM employees e
  WHERE salary IN (SELECT MAX(salary)
                     FROM employees
                     WHERE department_id = e.department_id);


/* 55. Write a query in SQL to display all the information of those employees who did not have any job in the past. */

SELECT *
  FROM employees
  WHERE employee_id NOT IN (SELECT employee_id
                              FROM job_history);