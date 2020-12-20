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
