supervised/task, unpervised/data driven, reinforcement/error
waterfall/sequence type- develop and then test, agile-continuous integration
timsort- merge sort and insertion sort
backpropagation(dl)-messenger telling network whether or not the net made a mistake when making prediction
difference between POST and GET method is that GET carries request parameter appended in URL string while 
POST carries request parameter in message body which makes it more secure way of transferring data from client t
cookies are employed to store user choices such as browsing session to trace the user preferences
Cache store online page resources during a browser for the long run purpose or to decrease the loading time
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