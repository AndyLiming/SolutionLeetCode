#include "head.hpp"
#include "solution.h"

//No 49 Group Anagrams
vector<vector<string>> solution::groupAnagrams(vector<string>& strs)
{
  vector<vector<string>> ans;
  if (strs.empty()) return ans;
  map<string, vector<string>> strMap;
  for (string s : strs) {
    string key = s;
    sort(key.begin(), key.end());
    strMap[key].push_back(s);
  }
  for (auto m : strMap) {
    sort(m.second.begin(), m.second.end());
    ans.push_back(m.second);
  }
  return ans;
}

//No 50 Pow(x, n)
double solution::myPow(double x, int n) {
  double res = 1.0;
  for (unsigned i = 0, an = abs(n);i <= 31;++i, an >>= 1, x *= x) {
    if (an & 1) {
      res *= x;
    }
  }
  return n < 0 ? 1 / res : res;
}

//No 54 Spiral Matrix
vector<int> solution::spiralOrder(vector<vector<int>>& matrix)
{
  vector<int> ans;
  if (matrix.size() == 0) return ans;
  int count = 1;
  int row = matrix.size();
  int col = matrix[0].size();
  int size = row*col;
  int x = 0, y = -1;
  while (count <= size) {
    vector<int>::iterator it;
    for (int j = y + 1;j < col;++j) {
      it = find(ans.begin(), ans.end(), matrix[x][j]);
      if (it == ans.end())
      {
        ans.push_back(matrix[x][j]);
        count++;
      }
      else
      {
        break;
      }
      y = j;
    }
    for (int i = x + 1; i<row; ++i)
    {
      it = find(ans.begin(), ans.end(), matrix[i][y]);
      if (it == ans.end())
      {
        ans.push_back(matrix[i][y]);
        count++;
      }
      else
      {
        break;
      }
      x = i;
    }
    for (int j = y - 1; j >= 0; --j)
    {
      it = find(ans.begin(), ans.end(), matrix[x][j]);
      if (it == ans.end())
      {
        ans.push_back(matrix[x][j]);
        count++;
      }
      else
      {
        break;
      }
      y = j;
    }
    for (int i = x - 1; i >= 0; --i)
    {
      it = find(ans.begin(), ans.end(), matrix[i][y]);
      if (it == ans.end())
      {
        ans.push_back(matrix[i][y]);
        count++;
      }
      else
      {
        break;
      }
      x = i;
    }

  }
  return ans;
}
//No 55 Jump Game
bool solution::canJump(vector<int>& nums)
{
  int end = 0;
  for (int i = 0;i < nums.size();++i) {
    if (i <= end) {
      end = max(end, i + nums[i]);
    }
    else break;
  }
  if (end >= nums.size() - 1) return true;
  return false;
}
//No 56 merge interval
static bool compInterval(Interval x, Interval y) {
  return x.start < y.start;
}
vector<Interval> solution::merge(vector<Interval>& intervals)
{
  if (intervals.size() < 1) return intervals;
  sort(intervals.begin(), intervals.end(), compInterval);
  int i = 0;
  while (i < intervals.size() - 1) {
    if (intervals[i].end < intervals[i + 1].start) i++;
    else {
      if (intervals[i].end < intervals[i + 1].end) {
        intervals[i].end = intervals[i + 1].end;
      }
      intervals.erase(intervals.begin() + i + 1);

    }
  }
  return intervals;
}
//No 59 Spiral Matrix II
vector<vector<int>> solution::generateMatrix(int n)
{
  vector<vector<int>> ans(n, vector<int>(n, 0));
  int dirs[4][2] = { {0, 1}, {1,0},{0,-1},{-1,0} };
  int cur = 1;
  int x = 0, y = 0, dir = 0;
  while (cur <= n*n) {
    ans[x][y] = cur++;
    int nx = x + dirs[dir][0];
    int ny = y + dirs[dir][1];
    if (nx < 0 || nx >= n || ny < 0 || ny >= n || ans[nx][ny] != 0) {
      dir = (dir + 1) % 4;
      x += dirs[dir][0];
      y += dirs[dir][1];
    }
    else {
      x = nx;
      y = ny;
    }
  }
  return ans;
}
//No 60 Permutation Sequence
string solution::getPermutation(int n, int k)
{
  string res;
  string num = "123456789";
  vector<int> fact(n, 1);
  for (int i = 1;i < n;++i) {
    fact[i] = fact[i - 1] * i;
  }
  --k;
  for (int i = n;i >= 1;--i) {
    int index = k / fact[i - 1];
    k = k%fact[i - 1];
    res.push_back(num[index]);
    num.erase(index, 1);
  }
  return res;
}
//No 61 Rotate List
ListNode * solution::rotateRight(ListNode * head, int k)
{
  if (!head) return nullptr;
  ListNode *p = head;
  int n = 1;
  while (p->next) {
    p = p->next;
    n++;
  }
  p->next = head;
  for (int i = 0;i < n - k%n;i++) {
    p = p->next;
  }
  ListNode * t = p;
  p = p->next;
  t->next = NULL;
  return p;
}
//No 62 Unique Paths
int solution::uniquePaths(int m, int n)
{
  vector<vector<int>> dp(n, vector<int>(m,1));
  for (int i = 1;i < n;++i) {
    for (int j = 1;j < m;++j) {
      dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
    }
  }
  return dp[n - 1][m - 1];
}
//No 63 Unique Paths II
int solution::uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid)
{
  int n= obstacleGrid.size(), m = obstacleGrid[0].size();
  vector<vector<int>> dp(n, vector<int>(m, 0));
  for (int i = 0;i < n;++i) {
    for (int j = 0;j < m;++j) {
      if (obstacleGrid[i][j] == 1) dp[i][j] = 0;
      else {
        if (i == 0 && j == 0) dp[i][j] = 1;
        else if (i == 0 && j > 0) dp[i][j] = dp[i][j-1];
        else if (i > 0 && j == 0) dp[i][j] = dp[i-1][j];
        else dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
      }
    }
  }
  return dp[n - 1][m - 1];
}
//No 64 Minimum Path Sum
int solution::minPathSum(vector<vector<int>>& grid)
{
  int n = grid.size(), m = grid[0].size();
  vector<vector<int>> dp(n, vector<int>(m, 0));
  for (int i = 0;i < n;++i) {
    for (int j = 0;j < m;++j) {
      if (i == 0 && j == 0) dp[i][j] = grid[i][j];
      else if(i == 0 && j > 0) dp[i][j] = dp[i][j-1] + grid[i][j];
      else if(i > 0 && j == 0) dp[i][j] = dp[i-1][j] + grid[i][j];
      else dp[i][j] = min(dp[i - 1][j],dp[i][j-1]) + grid[i][j];
    }
  }
  return dp[n - 1][m - 1];
}
//No 71 Simplify Path
string solution::simplifyPath(string path)
{
  string res;
  for (int i = 0, j, k; i < path.length(); i = j)
  {
    if (path[i] == '/')
    {
      for (j = i; j < path.length() && path[j] == '/'; ++j);
    }
    else
    {
      for (j = i; j < path.length() && path[j] != '/'; ++j);
      string fraction = path.substr(i, j - i);
      if (fraction == "..")
      {
        for (k = res.length() - 1; k >= 0 && res[k] != '/'; --k);
        if (k >= 0)
        {
          res.resize(k);
        }
      }
      else if (fraction != ".")
      {
        res.append("/").append(fraction);
      }
    }
  }
  return res.empty() ? "/" : res;
}
//No 73 Set Matrix Zeroes
void solution::setZeroes(vector<vector<int>>& matrix)
{
  if (matrix.size() == 0 || matrix[0].size() == 0) return;
  bool firstRowFlag = false, firstColFlag = false;
  for (int i = 0;i < matrix.size();++i) {
    if (matrix[i][0] == 0) {
      firstColFlag = true;
      break;
    }
  }
  for (int j = 0;j < matrix[0].size();++j) {
    if (matrix[0][j] == 0) {
      firstRowFlag = true;
      break;
    }
  }
  for (int i = 1;i < matrix.size();++i) {
    for (int j = 1;j < matrix[0].size();++j) {
      if (matrix[i][j] == 0) {
        matrix[i][0] = 0;
        matrix[0][j] = 0;
      }
    }
  }
  for (int i = 1;i < matrix.size();++i) {
    for (int j = 1;j < matrix[0].size();++j) {
      if(matrix[i][0]==0||matrix[0][j]==0){
        matrix[i][j] = 0;
      }
    }
  }
  if (firstColFlag) {
    for (int i = 0;i < matrix.size();++i) {
      matrix[i][0] = 0;
    }
  }
  if (firstRowFlag) {
    for (int j = 0;j < matrix[0].size();++j) {
      matrix[0][j] = 0;
    }
  }
}

bool solution::searchMatrix(vector<vector<int>>& matrix, int target)
{
  if (matrix.empty() || matrix[0].empty()) return false;
  if (target<matrix[0][0] || target>matrix[matrix.size() - 1][matrix[0].size() - 1]) return false;
  int fc = 0, fr = 0, lc = matrix[0].size() - 1, lr = matrix.size() - 1;
  int midc, midr;
  while (fr <= lr) {
    midr = (fr + lr) / 2;
    if (target == matrix[midr][0]) return true;
    if (target > matrix[midr][0]) fr = midr+1;
    if (target < matrix[midr][0]) lr = midr-1;
  }
  midr = lr;
  while (fc <= lc) {
    midc = (fc + lc) / 2;
    if (target == matrix[midr][midc]) return true;
    if (target > matrix[midr][midc]) fc = midc+1;
    if (target < matrix[midr][midc]) lc = midc-1;
  }
  return false;
}
//No 75 Sort Colors
void solution::sortColors(vector<int>& nums)
{
  if (nums.empty()) return;
  int red = 0, blue = nums.size() - 1;
  for (int i = 0;i <= blue;++i) {
    if (nums[i] == 0) {
      int temp = nums[i];
      nums[i] = nums[red];
      nums[red] = temp;
      red++;
    }
    else if (nums[i] == 2) {
      int temp = nums[i];
      nums[i] = nums[blue];
      nums[blue] = temp;
      blue--;
      i--;
    }
  }
}
//No 77 Combinations
vector<vector<int>> solution::combine(int n, int k)
{
  vector<vector<int>> ans, ans_2;
  if (k > n || k < 0) return ans;
  if (k == 0) {
    ans.push_back(vector<int>());
    return ans;
  }
  ans = combine(n - 1, k - 1);
  for (int i = 0;i < ans.size();++i) {
    ans[i].push_back(n);
  }
  ans_2 = combine(n - 1, k);
  for (int i = 0;i < ans_2.size();++i) {
    ans.push_back(ans_2[i]);
  }
  return ans;
}
//No 78 Subsets
vector<vector<int>> solution::subsets(vector<int>& nums)
{
  vector<vector<int>> ans;
  ans.push_back(vector<int>());
  sort(nums.begin(), nums.end());
  for (int i = 0;i < nums.size();++i) {
    int size = ans.size();
    for (int j = 0;j < size;++j) {
      ans.push_back(ans[j]);
      ans.back().push_back(nums[i]);
    }
  }
  return ans;
}
//No 79 Word Search
bool solution::exist(vector<vector<char>>& board, string word)
{
  if (word.length() == 0) return true;
  if (board.size() == 0 || board[0].size() == 0) return false;
  int rows = board.size();
  int cols = board[0].size();
  vector<vector<bool>> enable(rows, vector<bool>(cols, 1));
  for (int i = 0;i < rows;++i) {
    for (int j = 0;j < cols;++j) {
      if (board[i][j] == word[0] && exploreWordSearch(i, j, enable, 0, board, word)) return true;
    }
  }
  return false;
}

/* used in No 79 Word Search */
bool solution::exploreWordSearch(int row, int col, vector<vector<bool>>& enable, int position, const vector<vector<char>>& board, const string word)
{
  if (position == word.length() - 1) return true;
  enable[row][col] = 1;
  bool res = false;
  if (row > 0 && enable[row - 1][col] == 1 && board[row - 1][col] == word[position + 1]) {
    res = res || exploreWordSearch(row - 1, col, enable, position + 1, board, word);
  }
  if (row < board.size()-1 && enable[row + 1][col] == 1 && board[row + 1][col] == word[position + 1]) {
    res = res || exploreWordSearch(row + 1, col, enable, position + 1, board, word);
  }
  if (col > 0 && enable[row][col-1] == 1 && board[row][col-1] == word[position + 1]) {
    res = res || exploreWordSearch(row, col-1, enable, position + 1, board, word);
  }
  if (col < board[0].size()-1 && enable[row][col+1] == 1 && board[row][col+1] == word[position + 1]) {
    res = res || exploreWordSearch(row, col+1, enable, position + 1, board, word);
  }
  enable[row][col] = 0;
  return res;
}

//No 80 Remove Duplicates from Sorted Array II
int solution::removeDuplicates(vector<int>& nums)
{
  if (nums.size() <= 2) return nums.size();
  for (int i = 2;i < nums.size();) {
    if (nums[i] == nums[i - 1] && nums[i] == nums[i - 2]) {
      nums.erase(nums.begin() + i);
    }
    else ++i;
  }
  return nums.size();
}
//No 81 Search in Rotated Sorted Array II
bool solution::search(vector<int>& nums, int target)
{
  if(nums.size()==0) return false;
  int start = 0, end = nums.size() - 1;
  while (start <= end) {
    int mid = (start + end) / 2;
    if (nums[mid] == target) return true;
    if (nums[mid] > nums[start]) {
      if (nums[start] <= target&&target < nums[mid]) {
        end = mid - 1;
      }
      else {
        start = mid + 1;
      }
    }
    else if (nums[mid] < nums[start]) {
      if (nums[mid] < target&&target < nums[start]) {
        start = mid + 1;
      }
      else {
        end = mid - 1;
      }
    }
    else start++;
  }
  return false;
}
//No 82 Remove Duplicates from Sorted List II
ListNode * solution::deleteDuplicates(ListNode * head)
{
  if (head == nullptr || head->next == nullptr) return head;
  ListNode *helper = new ListNode(0);
  helper->next = head;
  ListNode *pre = helper;
  ListNode *cur = head;
  while (cur != nullptr) {
    while (cur->next != nullptr && pre->next->val == cur->next->val) {
      cur = cur->next;
    }
    if (pre->next == cur) {
      pre = cur;
    }
    else {
      pre->next = cur->next;
    }
    cur = cur->next;
  }
  return helper->next;
}
//No 84 Largest Rectangle in Histogram
int solution::largestRectangleArea(vector<int>& heights)
{
  int res = 0;
  stack<int> st;
  heights.push_back(0);
  for (int i = 0;i < heights.size();++i) {
    if (st.empty() || heights[st.top()] < heights[i]) {
      st.push(i);
    }
    else {
      int cur = st.top();
      st.pop();
      res = max(res, heights[cur] * (st.empty() ? i : (i - st.top() - 1)));
      --i;
    }
  }
  return res;
}

//No 85 Maximal Rectangle
int solution::maximalRectangle(vector<vector<char>>& matrix)
{
  if (matrix.empty() || matrix[0].empty()) return 0;
  int res = 0;
  int m = matrix.size(), n = matrix[0].size();
  vector<int> height(n + 1, 0);
  for (int i = 0;i < m;++i) {
    stack<int> st;
    for (int j = 0;j < n+1;++j) {
      if (j < n) {
        height[j] = matrix[i][j] == '1' ? height[j] + 1 : 0;
      }
      while (!st.empty() && height[st.top()] >= height[j]) {
        int cur = st.top();
        st.pop();
        res = max(res, height[cur] * (st.empty() ? j : (j - st.top() - 1)));
      }
      st.push(j);
    }
  }
  return res;
}
//No 86 Partition List
ListNode * solution::partition(ListNode * head, int x)
{
  ListNode* dummy = new ListNode(-1);
  dummy->next = head;
  ListNode *pre = dummy, *cur = head;
  while (pre->next && pre->next->val < x) pre = pre->next;
  cur = pre;
  while (cur->next) {
    if (cur->next->val < x) {
      ListNode *tmp = cur->next;
      cur->next = tmp->next;
      tmp->next = pre->next;
      pre->next = tmp;
      pre = pre->next;
    }
    else {
      cur = cur->next;
    }
  }
  return dummy->next;
}
//No 89 Gray Code
vector<int> solution::grayCode(int n)
{
  vector<int> ans;
  for (int i = 0;i < pow(2, n);++i) {
    ans.push_back((i/2)^i);
  }
  return ans;
}
//No 90 Subsets II
vector<vector<int>> solution::subsetsWithDup(vector<int>& nums)
{
  vector<vector<int>> ans;
  ans.push_back(vector<int>());
  if (nums.empty()) return ans;
  sort(nums.begin(), nums.end());
  for (int i = 0;i < nums.size();) {
    int count = 1;
    while (i + count < nums.size() && nums[i] == nums[i + count])
      count++;
    int size = ans.size();
    for (int j = 0;j < size;++j) {
      vector<int> tmp = ans[j];
      for (int k = 0;k < count;++k) {
        tmp.push_back(nums[i]);
        ans.push_back(tmp);
      }   
    }
    i += count;
  }
  return ans;
}
//No 91 Decode Ways
int solution::numDecodings(string s)
{
  if (s.empty()||s.size() > 1 && s[0]=='0') return 0;
  vector<int> dp(s.size() + 1, 0);
  dp[0] = 1;
  for (int i = 1;i < dp.size();++i) {
    dp[i] = (s[i - 1] == '0') ? 0 : dp[i - 1];
    if (i > 1 && (s[i - 2] == '1' || (s[i - 2] == '2'&&s[i - 1] <= '6'))) {
      dp[i] += dp[i - 2];
    }
  }
  return dp.back();
}
//No 92 Reverse Linked List II
ListNode * solution::reverseBetween(ListNode * head, int m, int n)
{
  ListNode * dummy = new ListNode(-1);
  dummy->next = head;
  ListNode *cur = dummy, *pre, *front=nullptr, *last;
  for (int i = 0;i < m-1;++i) {
    cur = cur->next;
  }
  pre = cur;
  last = cur->next;
  for (int i = m;i <= n;++i) {
    cur = pre->next;
    pre->next = cur->next;
    cur->next = front;
    front = cur;
  }
  cur = pre->next;
  pre->next = front;
  last->next = cur;
  return dummy->next;
}
//No 93 Restore IP Addresses
vector<string> solution::restoreIpAddresses(string s)
{
  vector<string> ans;
  restoreIpDfs(s, ans, 0, "");
  return ans;
}

 /* used in 93 Restore IP Addresses*/
void solution::restoreIpDfs(string s, vector<string>& ans, int dotNum, string partStr)
{
  if (dotNum > 4) return;
  if (dotNum == 4 && s.size() == 0) {
    ans.push_back(partStr);
    return;
  }
  if (dotNum <= 3) {
    for (int i = 1;i <= 3;++i) {
      if (s.size() >= i) {
        if (stoi(s.substr(0, i)) > 255) continue;
        if (i > 1 && s[0] == '0') continue;
        string partStrLocal = partStr;
        if (dotNum > 0) partStrLocal += '.';
        restoreIpDfs(s.substr(i), ans, dotNum + 1, partStrLocal + s.substr(0, i));
      }
    }
  }
}

//No 94 Binary Tree Inorder Traversal
vector<int> solution::inorderTraversal(TreeNode * root)
{
  vector<int> ans;
  stack<TreeNode*> st;
  TreeNode *p = root;
  while (p || !st.empty()) {
    while (p) {
      st.push(p);
      p = p->left;
    }
    p = st.top();
    st.pop();
    ans.push_back(p->val);
    p = p->right;
  }
  return ans;
}
//No 95 Unique Binary Search Trees II
//二分搜索树 中序遍历单调递增
//左节点存在，一定小于根；右节点存在，一定大于根
vector<TreeNode*> solution::generateTrees(int n)
{
  if (n == 0) return vector<TreeNode*>();
  return generateTreesDfs(1,n+1);
}

 /* used in 95 Unique Binary Search Trees II*/
vector<TreeNode*> solution::generateTreesDfs(int left, int right)
{
 vector<TreeNode*> subTree;
 if (left >= right) {
   subTree.push_back(NULL);
   return subTree;
 }
 if (left == right - 1) {
   subTree.push_back(new TreeNode(left));
   return subTree;
 }
 for (int i = left;i < right;++i) {
   vector<TreeNode*> leftSubTree = generateTreesDfs(left, i);
   vector<TreeNode*> rightSubTree = generateTreesDfs(i+1, right);
   for (int j = 0;j < leftSubTree.size();++j) {
     for (int k = 0;k < rightSubTree.size();++k) {
       TreeNode *root = new TreeNode(i);
       root->left = leftSubTree[j];
       root->right = rightSubTree[k];
       subTree.push_back(root);
     }
   }
 }
 return subTree;
}

//No 96 Unique Binary Search Trees
int solution::numTrees(int n)
{
  if (n <= 1) return 1;
  vector<int> dp(n + 1, 0);
  dp[0] = 1;
  dp[1] = 1;
  for (int i = 2;i < n + 1;++i) {
    for (int j = 0;j < i;++j) {
      dp[i] += dp[j] * dp[i - j-1];
    }
  }
  return dp[n];
}
//No 98 Validate Binary Search Tree
bool solution::isValidBST(TreeNode * root)
{
  if (root==NULL) return true;
  vector<int> vals;
  inorderBinaryTree(root, vals);
  for (int i = 1;i < vals.size();++i) {
    if (vals[i - 1] >= vals[i]) return false;
  }
  return true;
}

 /*used in 98 Validate Binary Search Tree*/
void solution::inorderBinaryTree(TreeNode * root, vector<int>& vals)
{
  if (!root) return;
  inorderBinaryTree(root->left, vals);
  vals.push_back(root->val);
  inorderBinaryTree(root->right, vals);
}
//No 102 Binary Tree Level Order Traversal
vector<vector<int>> solution::levelOrder(TreeNode * root)
{
  vector<vector<int>> ans;
  if (!root) return ans;
  queue<TreeNode*> q;
  q.push(root);
  while (!q.empty()) {
    vector<int> oneLevel;
    int size = q.size();
    for (int i = 0;i < size;++i) {
      TreeNode *node = q.front();
      q.pop();
      oneLevel.push_back(node->val);
      if (node->left) q.push(node->left);
      if (node->right) q.push(node->right);
    }
    ans.push_back(oneLevel);
  }
  return ans;
}
//No 103 Binary Tree Zigzag Level Order Traversal
vector<vector<int>> solution::zigzagLevelOrder(TreeNode * root)
{
  vector<vector<int>> ans;
  if (!root) return ans;
  stack<TreeNode*> s1;
  stack<TreeNode*> s2;
  s1.push(root);
  vector<int> level;
  while (!s1.empty() || !s2.empty()) {
    while (!s1.empty()) {
      TreeNode *cur = s1.top();
      s1.pop();
      level.push_back(cur->val);
      if (cur->left) s2.push(cur->left);
      if (cur->right) s2.push(cur->right);
    }
    if (!level.empty()) ans.push_back(level);
    level.clear();
    while (!s2.empty()) {
      TreeNode *cur = s2.top();
      s2.pop();
      level.push_back(cur->val);
      if (cur->right) s1.push(cur->right);
      if (cur->left) s1.push(cur->left); 
    }
    if (!level.empty()) ans.push_back(level);
    level.clear();
  }
  return ans;
}

//No 105 Construct Binary Tree from Preorder and Inorder Traversal
TreeNode * solution::buildTree_preIn(vector<int>& preorder, vector<int>& inorder)
{
  return buildTreeRes_preIn(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
}

TreeNode * solution::buildTreeRes_preIn(vector<int>& preorder, int preStart, int preEnd, vector<int>& inorder, int inStart, int inEnd)
{
  if (preStart > preEnd || inStart > inEnd) return NULL;
  TreeNode * root = new TreeNode(preorder[preStart]);
  int rootIndex = inStart;
  for (rootIndex = inStart;rootIndex <= inEnd;++rootIndex) {
    if (inorder[rootIndex] == root->val) break;
  }
  int leftNum = rootIndex - inStart;
  root->left = buildTreeRes_preIn(preorder, preStart + 1, preStart + leftNum, inorder, inStart, rootIndex - 1);
  root->right = buildTreeRes_preIn(preorder, preStart + leftNum + 1, preEnd, inorder, rootIndex + 1, inEnd);
  return root;
}

//No 106 Construct Binary Tree from Inorder and Postorder Traversal
TreeNode * solution::buildTree_inPos(vector<int>& inorder, vector<int>& postorder)
{
  return buildTreeRes_inPos(inorder,0, inorder.size()-1,postorder,0,postorder.size()-1);
}

TreeNode * solution::buildTreeRes_inPos(vector<int>& inorder, int inStart, int inEnd, vector<int>& postorder, int posStart, int posEnd)
{
  if (inStart > inEnd || posStart > posEnd) return NULL;
  TreeNode * root = new TreeNode(postorder[posEnd]);
  int rootIndex = inStart;
  for (rootIndex = inStart;rootIndex <= inEnd;++rootIndex) {
    if (inorder[rootIndex] == root->val) break;
  }
  int leftNum = rootIndex - inStart;
  root->left = buildTreeRes_inPos(inorder, inStart, rootIndex-1, postorder, posStart, posStart+leftNum-1);
  root->right= buildTreeRes_inPos(inorder, rootIndex + 1, inEnd, postorder, posStart + leftNum, posEnd - 1);
  return root;
}



//No 109 Convert Sorted List to Binary Search Tree
TreeNode * solution::sortedListToBST(ListNode * head)
{
  int len = 0;
  ListNode *p = head;
  while (p) {
    len++;
    p = p->next;
  }
  return buildBST(head, 0, len - 1);
}

TreeNode * solution::buildBST(ListNode *& list, int start, int end)
{
  if (start > end) return NULL;
  int mid = (start + end) / 2;
  TreeNode *left = buildBST(list, start, mid - 1);
  TreeNode * root = new TreeNode(list->val);
  root->left = left;
  list = list->next;
  root->right = buildBST(list, mid + 1, end);
  return root;
}

//No 113 Path Sum II
vector<vector<int>> solution::pathSum(TreeNode * root, int sum)
{
  vector<vector<int>> ans;
  if (root == NULL) return ans;
  vector<int> tmp;
  pathSumDfs(ans, tmp, root, sum);
  return ans;
}

void solution::pathSumDfs(vector<vector<int>> ans, vector<int> tmp, TreeNode * root, int sum)
{
  if (root == NULL) {
    tmp.push_back(0);
    return;
  }
  tmp.push_back(root->val);
  if (root->left == NULL && root->right == NULL) {
    if (root->val == sum) ans.push_back(tmp);
    return;
  }
  if(root->left != NULL) {
    pathSumDfs(ans, tmp, root->left, sum - root->val);
    tmp.pop_back();
  }
  if (root->right != NULL) {
    pathSumDfs(ans, tmp, root->right, sum - root->val);
    tmp.pop_back();
  }
}


//No 114 Flatten Binary Tree to Linked List
void solution::flatten(TreeNode * root)
{
  if (!root) return;
  if (root->left) flatten(root->left);
  if (root->right) flatten(root->right);
  TreeNode* tmp = root->right;
  root->right = root->left;
  root->left = NULL;
  TreeNode* p = root;
  while (p->right) p = p->right;
  p->right = tmp;
}

//No 116 Populating Next Right Pointers in Each Node
void solution::connect(TreeLinkNode * root)
{
  if (!root) return;
  if (root->left) root->left->next = root->right;
  if (root->right) root->right->next = root->next ? root->next->left : NULL;
  connect(root->left);
  connect(root->right);

  //非递归解
  /*
  if(!root) return;
  queue<TreeLinkNode*> q;
        q.push(root);
        q.push(NULL);
        while (true) {
            TreeLinkNode *cur = q.front();
            q.pop();
            if (cur) {
                cur->next = q.front();
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            } 
            else {
                if (q.size() == 0 || q.front() == NULL) return;
                q.push(NULL);
            }
        }
  */

  // space O(1)

  //if (!root) return;
  //TreeLinkNode *start = root, *cur = NULL;
  //while (start->left) {
  //  cur = start;
  //  while (cur) {
  //    cur->left->next = cur->right;
  //    if (cur->next) cur->right->next = cur->next->left;
  //    cur = cur->next;
  //  }
  //  start = start->left;
  //}
}

//No 117 Populating Next Right Pointers in Each Node II
void solution::connect_nonPerfect(TreeLinkNode * root)
{
  if (!root) return;
  TreeLinkNode* dummy = new TreeLinkNode(0);
  TreeLinkNode* t = dummy;
  while (root) {
    if (root->left) {
      t->next = root->left;
      t = t->next;
    }
    if (root->right) {
      t->next = root->right;
      t = t->next;
    }
    root = root->next;
    if (!root) {
      t = dummy;
      root = dummy->next;
      dummy->next = NULL;
    }
  }
}

//No 120 Triangle
int solution::minimumTotal(vector<vector<int>>& triangle)
{
  int rows = triangle.size();
  vector<int> dp(triangle.back());
  for (int i = rows - 2;i >= 0;--i) {
    for (int j = 0;j <= i;++j) {
      dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j];
    }
  }
  return dp[0];
}

//No 127 Word Ladder
int solution::ladderLength(string beginWord, string endWord, vector<string>& wordList)
{
  unordered_set<string> dict(wordList.begin(), wordList.end());
  if (dict.find(endWord)==dict.end()) return 0;
  int ans = 0;
  unordered_set<string> head;
  unordered_set<string> tail;
  head.insert(beginWord);tail.insert(endWord);
  while (!head.empty()|| !tail.empty()) {
    if (head.size() < tail.size()) swap(head, tail);
    ans++;
    unordered_set<string> tmp;
    for (string w : head){
      for (int i = 0;i<endWord.size();++i){
        char ctmp = w[i];
        for (char j = 'a';j <= 'z';++j){
          w[i] = j;
          if (tail.count(w) != 0) return ans + 1;
          if (dict.count(w) == 0) continue;
          tmp.insert(w);
          dict.erase(w); 
        }
        w[i] = ctmp; 
      }
    }
    swap(head, tmp);
  }
  return 0;
}

//No 129 Sum Root to Leaf Numbers
int solution::sumNumbers(TreeNode * root)
{
  return sumNumbersRes(root, 0);
}

int solution::sumNumbersRes(TreeNode * root, int curSum)
{
  if (!root) return 0;
  int valRoot = root->val;
  if (!root->left && !root->right) return curSum*10+valRoot;
  return sumNumbersRes(root->left, curSum * 10 + valRoot) + sumNumbersRes(root->right, curSum * 10 + valRoot);
}

//No 130 Surrounded Regions
void solution::solveSurronded(vector<vector<char>>& board)
{
  for (int i = 0;i < board.size();++i) {
    for (int j = 0;j < board[i].size();++j) {
      if ((i == 0 || i == board.size() - 1 || j == 0 || j == board[i].size() - 1) && board[i][j] == 'O')
        solveSurrondedDfs(board, i, j);
    }
  }

  for (int i = 0;i < board.size();++i) {
    for (int j = 0;j < board[i].size();++j) {
      if (board[i][j] == 'O') board[i][j] = 'X';
      if (board[i][j] == 'A') board[i][j] = 'O';
    }
  }
}
void solution::solveSurrondedDfs(vector<vector<char>>& board, int i, int j)
{
  if (board[i][j] == 'O') {
    board[i][j] = 'A';
    if (i > 0 && board[i - 1][j] == 'O') solveSurrondedDfs(board, i - 1, j);
    if (i < board.size()-1 && board[i + 1][j] == 'O') solveSurrondedDfs(board, i + 1, j);
    if (j > 0 && board[i][j-1] == 'O') solveSurrondedDfs(board, i, j-1);
    if (j < board[i].size() - 1 && board[i][j+1] == 'O') solveSurrondedDfs(board, i, j+1);
  }
}


//No 131 Palindrome Partitioning
vector<vector<string>> solution::partition(string s)
{
  vector<vector<string>> ans;
  vector<string> out;
  partitionDfs(ans, out, s, 0);
  return ans;
}
bool solution::isValidPali(string s, int start, int end)
{
  while (start < end) {
    if (s[start] != s[end]) return false;
    ++start;
    --end;
  }
  return true;
}
void solution::partitionDfs(vector<vector<string>>& ans, vector<string> out, string s, int start)
{
  if (start == s.size()) {
    ans.push_back(out);
    return;
  }
  for (int i = start;i < s.length();++i) {
    if (isValidPali(s, start,i)) {
      out.push_back(s.substr(start, i - start + 1));
      partitionDfs(ans, out, s, i + 1);
      out.pop_back();
    }
  }
}

//No 133 Clone Graph
UndirectedGraphNode * solution::cloneGraph(UndirectedGraphNode * node)
{
  unordered_map<int, UndirectedGraphNode*> umap;
  return cloneGraphHelper(node, umap);
}

UndirectedGraphNode * solution::cloneGraphHelper(UndirectedGraphNode * node, unordered_map<int, UndirectedGraphNode*>& umap)
{
  if (!node) return node;
  if (umap.count(node->label)) return umap[node->label];
  UndirectedGraphNode* newNode = new UndirectedGraphNode(node->label);
  umap[node->label] = newNode;
  for (int i = 0;i < node->neighbors.size();++i) {
    newNode->neighbors.push_back(cloneGraphHelper(node->neighbors[i], umap));
  }
  return newNode;
}

//No 134 Gas Station
int solution::canCompleteCircuit(vector<int>& gas, vector<int>& cost)
{
  int start=0, sum=0, total=0;
  for (int i = 0;i < gas.size();++i) {
    total += gas[i] - cost[i];
    sum += gas[i] - cost[i];
    if (sum < 0) {
      start = i + 1;
      sum = 0;
    }
  }
  return total < 0 ? -1 : start;
}
//No 137 Single Number II
int solution::singleNumber(vector<int>& nums)
{
  sort(nums.begin(), nums.end());
  if (nums[0] != nums[1]) return nums[0];
  for (int i = 1;i < nums.size() - 1;++i) {
    if (nums[i] != nums[i - 1] && nums[i] != nums[i + 1])
      return nums[i];
  }
  return nums.back();
}
//No 138 Copy List with Random Pointer
RandomListNode * solution::copyRandomList(RandomListNode * head)
{
  if (!head) return NULL;
  RandomListNode *ans = new RandomListNode(head->label);
  RandomListNode *node = ans;
  RandomListNode *cur = head->next;
  map<RandomListNode*, RandomListNode*> m;
  m[head] = ans;
  while (cur) {
    RandomListNode *tmp = new RandomListNode(cur->label);
    node->next = tmp;
    m[cur] = tmp;
    node = node->next;
    cur = cur->next;
    }
  node = ans;
  cur = head;
  while (node) {
    node->random = m[cur->random];
    node = node->next;
    cur = cur->next;
  }
  return ans;
}

//No 139 Word Break
bool solution::wordBreak(string s, vector<string>& wordDict)
{
  vector<bool> flags(s.length() + 1, false);
  unordered_set<string> dict(wordDict.begin(), wordDict.end());
  flags[0] = true;
  for (int i = 0;i < s.length() + 1;++i) {
    for (int j = i - 1;j >= 0;--j) {
      if (flags[j] && dict.count(s.substr(j, i - j)) != 0) {
        flags[i] = true;
        break;
      }
    }
  }
  return flags[s.length()];
}

//No 141 Linked List Cycle
bool solution::hasCycle(ListNode * head)
{
  if(!head||!head->next) return false;
  ListNode * fast = head, *slow = head;
  while (fast&&fast->next) {
    slow = slow->next;
    fast = fast->next->next;
    if (fast == slow) return true;
  }
  return false;
}

//No 142 Linked List Cycle II
ListNode * solution::detectCycle(ListNode * head)
{
  if (!head || !head->next) return NULL;
  ListNode * fast = head, *slow = head;
  while (fast&&fast->next) {
    slow = slow->next;
    fast = fast->next->next;
    if (fast == slow) break;
  }
  if (!fast || !fast->next) return NULL;
  slow = head;
  while (fast != slow) {
    slow = slow->next;
    fast = fast->next;
  }
  return slow;
}

//No 143 Reorder List
void solution::reorderList(ListNode * head)
{
  if (!head || !head->next) return;
  ListNode * fast = head, *slow = head;
  while (fast&&fast->next) {
    slow = slow->next;
    fast = fast->next->next;
  }
  ListNode* mid = slow->next;
  slow->next = NULL;
  ListNode* pre = NULL;
  while (mid) {
    ListNode* tmp = mid->next;
    mid->next = pre;
    pre = mid;
    mid = tmp;
  }
  ListNode *first = head;
  mid = pre;
  while (first&&mid) {
    ListNode *ftmp = first->next;
    ListNode *mtmp = mid->next;
    first->next = mid;
    first = ftmp;
    mid->next = first;
    mid = mtmp;
  }
}

//No 144 Binary Tree Preorder Traversal
vector<int> solution::preorderTraversal(TreeNode * root)
{
  //iteratively
  vector<int> ans;
  if (!root) return ans;
  stack<TreeNode *> s;
  s.push(root);
  while (!s.empty()) {
    TreeNode * t = s.top();
    s.pop();
    ans.push_back(t->val);
    if (t->right) s.push(t->right);
    if (t->left) s.push(t->left);
  }
  return ans;
  //Recursive 
  //vector<int> ans;
  //if (!root) return ans;
  //ans.push_back(root->val);
  //vector<int> left = preorderTraversal(root->left);
  //vector<int> right = preorderTraversal(root->right);
  //for (int i = 0;i < left.size();++i) {
  //  ans.push_back(left[i]);
  //}
  //for (int i = 0;i < right.size();++i) {
  //  ans.push_back(right[i]);
  //}
  //return ans;
}

//No 145 Binary Tree Postorder Traversal
vector<int> solution::postorderTraversal(TreeNode * root)
{
  //iteratively
  vector<int> ans;
  if (!root) return ans;
  stack<TreeNode *> s;
  deque<int> dq;
  s.push(root);
  while (!s.empty()) {
    TreeNode * t = s.top();
    s.pop();
    dq.push_front(t->val);
    if (t->left) s.push(t->left);
    if (t->right) s.push(t->right);
  }
  while (!dq.empty()) {
    ans.push_back(dq.front());
    dq.pop_front();
  }
  return ans;
  //Recursive
  /*vector<int> ans;
  if (!root) return ans;
  vector<int> left = postorderTraversal(root->left);
  vector<int> right = postorderTraversal(root->right);
  for (int i = 0;i < left.size();++i) {
    ans.push_back(left[i]);
  }
  for (int i = 0;i < right.size();++i) {
    ans.push_back(right[i]);
  }
  ans.push_back(root->val);
  return ans;*/
}

//No 147 Insertion Sort List
ListNode * solution::insertionSortList(ListNode * head)
{
  if (!head || !head->next) return head;
  ListNode *dummy = new ListNode(0);
  ListNode *cur = head;
  ListNode *pre = dummy;
  while (cur) {
    ListNode *next = cur->next;
    pre = dummy;
    while (pre->next&&pre->next->val <= cur->val) {
      pre = pre->next;
    }
    cur->next = pre->next;
    pre->next = cur;
    cur = next;
  }
  return dummy->next;
}

//No 148 Sort List
ListNode * solution::sortList(ListNode * head)
{
  if (!head||!head->next) return head;
  ListNode *fast = head, *slow = head,*pre=head;
  while (fast&&fast->next) {
    pre = slow;
    fast = fast->next->next;
    slow = slow->next;
  }
  pre->next = NULL;
  return mergeSortedList(sortList(head), sortList(slow));
}

ListNode * solution::mergeSortedList(ListNode * list1, ListNode * list2)
{
  if (!list1) return list2;
  if (!list2) return list1;
  if (list1->val < list2->val) {
    list1->next = mergeSortedList(list1->next, list2);
    return list1;
  }
  else {
    list2->next = mergeSortedList(list1, list2->next);
    return list2;
  }
}


//No 150 Evaluate Reverse Polish Notation
int solution::evalRPN(vector<string>& tokens)
{
  if (tokens.size() == 1) {
    return atoi(tokens[0].c_str());
  }
  stack<int> s;
  for (int i = 0;i < tokens.size();++i) {
    if (tokens[i] != "+" && tokens[i] != "-" && tokens[i] != "*" && tokens[i] != "/") {
      s.push(atoi(tokens[i].c_str()));
    }
    else {
      int m = s.top();
      s.pop();
      int n = s.top();
      s.pop();
      if (tokens[i] == "+") s.push(n + m);
      if (tokens[i] == "-") s.push(n - m);
      if (tokens[i] == "*") s.push(n * m);
      if (tokens[i] == "/") s.push(n / m);
    }
  }
  return s.top();
}

//No 151 Reverse Words in a String
void solution::reverseWords(string & s)
{
  istringstream is(s);
  string tmp;
  is >> s;
  while (is >> tmp) {
    s = tmp + " " + s;
  }
  if (!s.empty() && s[0] == ' ') s = "";
}

//No 152 Maximum Product Subarray
int solution::maxProduct(vector<int>& nums)
{
  long long maxPro = nums[0], preMax = nums[0],preMin=nums[0];
  long long curMax, curMin;
  if (nums.size() == 1) return maxPro > INT32_MAX ? INT32_MAX : (int)maxPro;
  for (int i = 1;i < nums.size();++i) {
    curMin = min(min(preMax*nums[i], preMin*nums[i]), (long long)nums[i]);
    curMax = max(max(preMax*nums[i], preMin*nums[i]), (long long)nums[i]);
    maxPro = max(curMax, maxPro);
    preMax = curMax;
    preMin = curMin;
  }
  return maxPro > INT32_MAX ? INT32_MAX : (int)maxPro;
}

//No 153 Find Minimum in Rotated Sorted Array
int solution::findMin(vector<int>& nums)
{
  if (nums.size() == 1) return nums[0];
  for (int i = 1;i < nums.size();++i) {
    if (nums[i] < nums[i - 1]) return nums[i];
  }
  return nums[0];
}

//No 160 Intersection of Two Linked Lists
ListNode * solution::getIntersectionNode(ListNode * headA, ListNode * headB)
{
  if (!headA || !headB) return NULL;
  int lenA = 0, lenB = 0;
  ListNode *helpA = headA, *helpB = headB;
  while (helpA) {
    helpA = helpA->next;
    lenA++;
  }
  while (helpB) {
    helpB = helpB->next;
    lenB++;
  }
  if (lenA > lenB) {
    for (int i = 0;i < lenA - lenB;++i) {
      headA = headA->next;
    }
  }
  else {
    for (int i = 0;i < lenB - lenA;++i) {
      headB = headB->next;
    }
  }
  while (headA && headB && headA != headB) {
    headA = headA->next;
    headB = headB->next;
  }
  return(headA&&headB) ? headA : NULL;
}

//No 162 Find Peak Element
int solution::findPeakElement(vector<int>& nums)
{
  if (nums.size() == 1) return 0;
  if (nums[0] > nums[1]) return 0;
  for (int i = 1;i < nums.size() - 1;++i) {
    if (nums[i] > nums[i - 1] && nums[i] > nums[i + 1]) return i;
  }
  return nums.size() - 1;
}

//No 165 Compare Version Numbers
int solution::compareVersion(string version1, string version2)
{
  istringstream v1(version1 + "."), v2(version2 + ".");
  char dot = '.';
  int d1 = 0, d2 = 0;
  while (v1.good() || v2.good()) {
    if (v1.good()) v1 >> d1 >> dot;
    if (v2.good()) v2 >> d2 >> dot;
    if (d1 > d2) return 1;
    else if (d1 < d2) return -1;
    d1 = d2 = 0;
  }
  return 0;
}

//No 166 Fraction to Recurring Decimal
string solution::fractionToDecimal(int numerator, int denominator)
{
  if (numerator == 0) return "0";
  string result;
  if (numerator < 0 ^ denominator < 0) result += '-';
  long long int n = numerator, d = denominator;
  n = abs(n);
  d = abs(d);
  result += to_string(n / d);
  long long int r = n%d;
  if (r == 0) return result;
  else result += '.';
  unordered_map<int, int>m;
  while (r) {
    if (m.find(r) != m.end()) {
      result.insert(m[r], 1, '(');
      result += ')';
      break;
    }
    m[r] = result.size();

    r *= 10;
    result += to_string(r / d);
    r = r%d;
  }
  return result;
}

//No 167 Two Sum II - Input array is sorted
vector<int> solution::twoSum(vector<int>& numbers, int target)
{
  vector<int> ans;
  int left = 0, right = numbers.size() - 1;
  int sum;
  while (left < right) {
    sum = numbers[left] + numbers[right];
    if (sum == target) {
      ans.push_back(left+1);
      ans.push_back(right+1);
      return ans;
    }
    if (sum > target) right--;
    if (sum < target) left++;
  }
}

//No 168 Excel Sheet Column Title
string solution::convertToTitle(int n)
{
  string ans = "";
  while (n)
  {
    ans = (char)((n - 1) % 26 + 'A') + ans;
    n = (n - 1) / 26;
  }
  return ans;
}

//No 169 Majority Element
int solution::majorityElement(vector<int>& nums)
{
  int elem = 0, count = 0;
  for (int i = 0;i < nums.size();++i) {
    if (count == 0) {
      elem = nums[i];
      count = 1;
    }
    else {
      if (elem == nums[i]) {
        count++;
      }
      else
        count--;
    }
  }
  return elem;
}

//No 171 Excel Sheet Column Number
int solution::titleToNumber(string s)
{
  int ans = 0;
  if (s.empty()) return ans;
  char tmp;
  for (int i = 0;i < s.length();++i) {
    tmp = s[i];
    ans = ans * 26 + (tmp - 'A' + 1);
  }
  return ans;
}

//No 172 Factorial Trailing Zeroes
int solution::trailingZeroes(int n)
{
  int ans = 0;
  while (n) {
    ans += n / 5;
    n /= 5;
  }
  return ans;
}

//No 189 Rotate Array
void solution::rotate(vector<int>& nums, int k)
{
  int len = nums.size();
  k = k%len;
  for (int i = 0;i < len-k;++i) {
    nums.push_back(nums.front());
    nums.erase(nums.begin());
  }
  return;
}

//No 190 Reverse Bits
uint32_t solution::reverseBits(uint32_t n)
{
  uint32_t ans=0;
  int count = 0;
  while (count < 32) {
    ans = ans << 1;
    if (n & 1) ans++;
    n =n>> 1;
    count++;
  }
  return ans;
}

//No 191 Number of 1 Bits
int solution::hammingWeight(uint32_t n)
{
  int a = 0;
  int count = 0;
  while (count < 32) {
    if (n & 1) a++;
    n = n >> 1;
    count++;
  }
  return a;
}
//No 198 Rob House
int solution::rob(vector<int>& nums)
{
  if (nums.empty()) return 0;
  if (nums.size() == 1) return nums[0];
  int a= 0, b = 0;
  for (int i = 0;i < nums.size();++i) {
    if (i % 2 == 0) a = max(a+nums[i],b);
    else b = max(b+nums[i],a);
  }
  return max(a, b);
}

//No 199 Binary Tree Right Side View
vector<int> solution::rightSideView(TreeNode * root)
{
  vector<int> ans;
  rightSVdfs(root, 0, ans);
  return ans;
}

void solution::rightSVdfs(TreeNode * root, int deep, vector<int> &ans)
{
  if (!root) return;
  if (deep >= ans.size()) ans.push_back(root->val);
  rightSVdfs(root->right, deep + 1, ans);
  rightSVdfs(root->left, deep + 1, ans);
}


//No 200 Number of Islands
int solution::numIslands(vector<vector<char>>& grid)
{
  if (grid.empty()) return 0;
  int num = 0;
  for (int i = 0;i < grid.size();++i) {
    for (int j = 0;j < grid[0].size();++j) {
      if (grid[i][j] == '1') {
        DfsNumIslands(grid, i, j);
        num++;
      }
    }
  }
  return num;
}

void solution::DfsNumIslands(vector<vector<char>>& grid, int row, int col)
{
  if (row < 0 || row >= grid.size() || col < 0 || col >= grid[0].size() || grid[row][col] == '0') return;
  grid[row][col] = '0';
  DfsNumIslands(grid, row + 1, col);
  DfsNumIslands(grid, row - 1, col);
  DfsNumIslands(grid, row, col + 1);
  DfsNumIslands(grid, row, col - 1);
}


//No 201 Bitwise AND of Numbers Range
int solution::rangeBitwiseAnd(int m, int n)
{
  int count = 0;
  while (n != m) {
    n >>= 1;
    m >>= 1;
    count++;
  }
  return (m << count);
}

//No 202 Happy Number
bool solution::isHappy(int n)
{
  set<int> squSumSet;
  while (n != 1) {
    int t = 0;
    while (n) {
      t = t + (n % 10)*(n % 10);
      n = n / 10;
    }
    n = t;
    if (squSumSet.count(t) != 0) break;
    else squSumSet.insert(t);
  }
  return n == 1;
}

//No 203 Remove Linked List Elements
ListNode * solution::removeElements(ListNode * head, int val)
{
  if (head == nullptr) return head;
  ListNode * helper = new ListNode(0);
  helper->next = head;
  ListNode *p = helper;
  while (p->next) {
    if (p->next->val == val) p->next = p->next->next;
    else p = p->next;
  }
  return helper->next;
}

//No 204 Count Primes
int solution::countPrimes(int n)
{
  if (n <= 2) return 0;
  vector<bool> notPrimes(n,false);
  int count = 1;
  for (int x = 3;x*x <n;x=x+2) {
    if (notPrimes[x]) {
      continue;
    }
    else{
      for (int j = x * x;j < n;j = j+2 * x) {
        notPrimes[j] = true;
      }
    }
  }
  for (int x = 3;x < n;x=x+2) {
    if (!notPrimes[x]) count++;
  }
  return count;
}

bool solution::isPrime(int x, vector<int> primes)
{
  for (int i = 0;primes[i]* primes[i] <= x;++i) {
    if (x%primes[i] == 0) return false;
  }
  return true;
}


//No 205 Isomorphic Strings
bool solution::isIsomorphic(string s, string t)
{
  if (s.size()!=t.size()) return false;
  unordered_map<char, vector<int>> hashS;
  unordered_map<char, vector<int>> hashT;
  for (int i = 0;i < s.size();++i) {
    hashS[s[i]].push_back(i);
    hashT[t[i]].push_back(i);
  }
  for (int i = 0;i < s.size();++i) {
    if (hashS[s[i]].size() == 1 && hashT[t[i]].size() == 1) continue;
    if (hashS[s[i]] != hashT[t[i]]) return false;
  }
  return true;
}

//No 206 Reverse Linked List
ListNode * solution::reverseList(ListNode * head)
{
  if (!head || !head->next) return head;
  ListNode *p = head;
  ListNode *q = head;
  p = p->next;
  while (p) {
    q->next = p->next;
    p->next = head;
    head = p;
    p = q->next;
  }
  return head;
}

//No 207 Course Schedule
bool solution::canFinish(int numCourses, vector<pair<int, int>>& prerequisites)
{
  vector<int> heads(numCourses, -1);
  vector<int> enDegree(numCourses, 0);
  vector<int> points, args;
  pair<int,int> p;
  int from, to, count = 0, len = prerequisites.size();
  for (int i = 0;i < len;++i) {
    p = prerequisites[i];
    from = p.second;
    to = p.first;
    ++enDegree[to];
    args.push_back(heads[from]);
    points.push_back(to);
    heads[from] = count++;
  }
  queue<int> q;
  for (int i = 0;i < numCourses;++i) {
    if (enDegree[i] == 0) q.push(i);
  }
  while (!q.empty()) {
    from = q.front();
    q.pop();
    to = heads[from];
    while (to != -1) {
      if (--enDegree[points[to]]==0) q.push(points[to]);
      to = args[to];
    }
  }
  for (int i = 0;i < numCourses;++i) {
    if (enDegree[i] > 0) return false;
  }
  return true;
}

//No 209 Minimum Size Subarray Sum
int solution::minSubArrayLen(int s, vector<int>& nums)
{
  if(nums.empty()) return 0;
  int left = 0, right = 0, sum = 0, res = nums.size() + 1;
  while (right<nums.size()) {
    while (sum < s&&right < nums.size()) {
      sum += nums[right++];
    }
    while (sum >= s) {
      res = min(res, right - left);
      sum -= nums[left++];
    }
  }
  return res == nums.size() + 1 ? 0 : res;
}

//No 210 Course Schedule II
vector<int> solution::findOrder(int numCourses, vector<pair<int, int>>& prerequisites)
{
  vector<int> heads(numCourses, -1);
  vector<int> enDegree(numCourses, 0);
  vector<int> points, args, ans;
  pair<int, int> p;
  int from, to, count = 0, len = prerequisites.size();
  for (int i = 0;i < len;++i) {
    p = prerequisites[i];
    from = p.second;
    to = p.first;
    ++enDegree[to];
    args.push_back(heads[from]);
    points.push_back(to);
    heads[from] = count++;
  }
  queue<int> q;
  for (int i = 0;i < numCourses;++i) {
    if (enDegree[i] == 0) q.push(i);
  }
  while (!q.empty()) {
    from = q.front();
    ans.push_back(from);
    q.pop();
    to = heads[from];
    while (to != -1) {
      if (--enDegree[points[to]] == 0) q.push(points[to]);
      to = args[to];
    }
  }
  for (int i = 0;i < numCourses;++i) {
    if (enDegree[i] > 0) return vector<int>();
  }
  return ans;
}

//No 213 House Robber II
int solution::rob2(vector<int>& nums)
{
  if (nums.size() == 0) return 0;
  if (nums.size() == 1) return nums[0];
  vector<int> tmp1(nums.begin(), nums.end() - 1);
  vector<int> tmp2(nums.begin() + 1, nums.end());
  return max(robPlan(tmp1), robPlan(tmp2));
}

int solution::robPlan(vector<int>& nums)
{
  vector<int> dp(nums.size(), 0);
  dp[0] = nums[0];
  dp[1] = max(nums[0],nums[1]);
  dp[2] = max(nums[1],nums[0]+nums[2]);
  for (int i = 3;i < nums.size();++i) {
    dp[i] = max(dp[i - 3] + nums[i - 1], dp[i - 2] + nums[i]);
  }
  return max(dp[nums.size() - 1], dp[nums.size() - 2]);
}


//implementation of KMP
int solution::KMP(string s, string t)
{
  vector<int> next(t.length(), -1);
  int k = -1;
  //calculate vector next
  for (int q = 1;q < t.length();++q) {
    while (k > -1 && t[k + 1] != t[q]) {
      k = next[k];
    }
    if (t[k + 1] == t[q]) {
      k++;
    }
    next[q] = k;
  }
  //for (int p = 0;p < next.size();++p) {
  //  cout << next[p] << " ";
  //}
  //kmp
  int i = 0,j = 0;
  int lenS = s.size(), lenT = t.size();
  while (i < lenS && j < lenT) {
    if (j == -1 || s[i] == t[j]) {
      i++;
      j++;
    }
    else {
      j = next[j];
    }
  }
  return j == t.length() ? i - j : -1;
}

//No 214 Shortest Palindrome: KMP based solution
string solution::shortestPalindrome(string s)
{
  string revS=s;
  reverse(revS.begin(), revS.end());
  string t = s+'#'+revS;
  vector<int>next(t.size(), -1);
  int k = -1;
  //calculate vector next
  for (int q = 1;q < t.length();++q) {
    while (k > -1 && t[k + 1] != t[q]) {
      k = next[k];
    }
    if (t[k + 1] == t[q]) {
      k++;
    }
    next[q] = k;
  }
  cout << next[t.size() - 1] << endl;
  string ans = revS.substr(0, revS.size() - next[t.size() - 1]-1) + s;
  return ans;
}

//No 215 Kth Largest Element in an Array
int solution::findKthLargest(vector<int>& nums, int k)
{
  sort(nums.begin(), nums.end());
  return nums[nums.size() - k];
}

//No 216 Combination Sum III
vector<vector<int>> solution::combinationSum3(int k, int n)
{
  vector<vector<int>> ans;
  vector<int> out;
  combinationSum3Dfs(k, n, 1, out, ans);
  return ans;
}

void solution::combinationSum3Dfs(int k, int n, int level, vector<int>& out, vector<vector<int>>& ans)//&
{
  if (n < 0||out.size()>k) return;
  if (n == 0 && out.size() == k) ans.push_back(out);
  for (int i = level;i <= 9;++i) {
    out.push_back(i);
    combinationSum3Dfs(k, n - i, i + 1, out, ans);
    out.pop_back();
  }
}

//No 217 Contains Duplicate
bool solution::containsDuplicate(vector<int>& nums)
{
  set<int>numSet;
  for (int i = 0;i < nums.size();++i) {
    if (numSet.find(nums[i]) != numSet.end()) return true;
    else {
      numSet.insert(nums[i]);
    }
  }
  return false;
}

//No 219 Contains Duplicate II
bool solution::containsNearbyDuplicate(vector<int>& nums, int k)
{
  map<int, int>numMap;
  for (int i = 0;i < nums.size();++i) {
    if (numMap.find(nums[i]) != numMap.end()) {
      if ((i - numMap[nums[i]]) <= k) return true;
      else numMap[nums[i]] = i;
    }
    else {
      numMap.insert(pair<int, int>(nums[i], i));//map must insert a pair<key_type,value_type>
    }
  }
  return false;
}

//No 220 Contains Duplicate III
bool solution::containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t)
{
  set<long> numSet;
  long lt = t;
  for (int i = 0;i < nums.size();++i) {
    if (i > k) numSet.erase(nums[i - k - 1]);
    set<long>::iterator it = numSet.lower_bound(nums[i] - lt);
    if (it != numSet.end() && (*it) - nums[i] <= lt) return true;
    numSet.insert(nums[i]);
  }
  return false;
}

//No 221 Maximal Square
int solution::maximalSquare(vector<vector<char>>& matrix)//must be square
{
  if(matrix.empty()) return 0;
  int ans = 0;
  vector<vector<int>> dp (matrix.size(), vector<int>(matrix[0].size(), 0));
  for (int i = 0;i < matrix.size();++i) {
    for (int j = 0;j < matrix[0].size();++j) {
      if (i > 0 && j > 0 && matrix[i][j] == '1') {
        dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
      }
      else {
        dp[i][j] = matrix[i][j] - '0';
      }
      ans = max(ans, dp[i][j]);
    }
  }
  return ans*ans;
}

//No 222 Count Complete Tree Nodes
int solution::countNodes(TreeNode * root)
{
  int hLeft = 0, hRight = 0;
  TreeNode *pLeft = root, *pRight = root;
  while (pLeft) {
    ++hLeft;
    pLeft = pLeft->left;
  }
  while (pLeft) {
    ++hRight;
    pRight = pRight->right;
  }
  if (hLeft == hRight) return pow(2, hLeft) - 1;
  return countNodes(root->left) + countNodes(root->right) + 1;
}
