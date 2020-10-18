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
	for (unsigned i = 0, an = abs(n); i <= 31; ++i, an >>= 1, x *= x) {
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
	int size = row * col;
	int x = 0, y = -1;
	while (count <= size) {
		vector<int>::iterator it;
		for (int j = y + 1; j < col; ++j) {
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
		for (int i = x + 1; i < row; ++i)
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
	for (int i = 0; i < nums.size(); ++i) {
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
vector<Interval> solution::merge(vector<Interval> & intervals)
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
	while (cur <= n * n) {
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
	for (int i = 1; i < n; ++i) {
		fact[i] = fact[i - 1] * i;
	}
	--k;
	for (int i = n; i >= 1; --i) {
		int index = k / fact[i - 1];
		k = k % fact[i - 1];
		res.push_back(num[index]);
		num.erase(index, 1);
	}
	return res;
}
//No 61 Rotate List
ListNode* solution::rotateRight(ListNode * head, int k)
{
	if (!head) return nullptr;
	ListNode* p = head;
	int n = 1;
	while (p->next) {
		p = p->next;
		n++;
	}
	p->next = head;
	for (int i = 0; i < n - k % n; i++) {
		p = p->next;
	}
	ListNode* t = p;
	p = p->next;
	t->next = NULL;
	return p;
}
//No 62 Unique Paths
int solution::uniquePaths(int m, int n)
{
	vector<vector<int>> dp(n, vector<int>(m, 1));
	for (int i = 1; i < n; ++i) {
		for (int j = 1; j < m; ++j) {
			dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
		}
	}
	return dp[n - 1][m - 1];
}
//No 63 Unique Paths II
int solution::uniquePathsWithObstacles(vector<vector<int>> & obstacleGrid)
{
	int n = obstacleGrid.size(), m = obstacleGrid[0].size();
	vector<vector<int>> dp(n, vector<int>(m, 0));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			if (obstacleGrid[i][j] == 1) dp[i][j] = 0;
			else {
				if (i == 0 && j == 0) dp[i][j] = 1;
				else if (i == 0 && j > 0) dp[i][j] = dp[i][j - 1];
				else if (i > 0 && j == 0) dp[i][j] = dp[i - 1][j];
				else dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
	}
	return dp[n - 1][m - 1];
}
//No 64 Minimum Path Sum
int solution::minPathSum(vector<vector<int>> & grid)
{
	int n = grid.size(), m = grid[0].size();
	vector<vector<int>> dp(n, vector<int>(m, 0));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			if (i == 0 && j == 0) dp[i][j] = grid[i][j];
			else if (i == 0 && j > 0) dp[i][j] = dp[i][j - 1] + grid[i][j];
			else if (i > 0 && j == 0) dp[i][j] = dp[i - 1][j] + grid[i][j];
			else dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
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
void solution::setZeroes(vector<vector<int>> & matrix)
{
	if (matrix.size() == 0 || matrix[0].size() == 0) return;
	bool firstRowFlag = false, firstColFlag = false;
	for (int i = 0; i < matrix.size(); ++i) {
		if (matrix[i][0] == 0) {
			firstColFlag = true;
			break;
		}
	}
	for (int j = 0; j < matrix[0].size(); ++j) {
		if (matrix[0][j] == 0) {
			firstRowFlag = true;
			break;
		}
	}
	for (int i = 1; i < matrix.size(); ++i) {
		for (int j = 1; j < matrix[0].size(); ++j) {
			if (matrix[i][j] == 0) {
				matrix[i][0] = 0;
				matrix[0][j] = 0;
			}
		}
	}
	for (int i = 1; i < matrix.size(); ++i) {
		for (int j = 1; j < matrix[0].size(); ++j) {
			if (matrix[i][0] == 0 || matrix[0][j] == 0) {
				matrix[i][j] = 0;
			}
		}
	}
	if (firstColFlag) {
		for (int i = 0; i < matrix.size(); ++i) {
			matrix[i][0] = 0;
		}
	}
	if (firstRowFlag) {
		for (int j = 0; j < matrix[0].size(); ++j) {
			matrix[0][j] = 0;
		}
	}
}

bool solution::searchMatrix(vector<vector<int>> & matrix, int target)
{
	if (matrix.empty() || matrix[0].empty()) return false;
	if (target<matrix[0][0] || target>matrix[matrix.size() - 1][matrix[0].size() - 1]) return false;
	int fc = 0, fr = 0, lc = matrix[0].size() - 1, lr = matrix.size() - 1;
	int midc, midr;
	while (fr <= lr) {
		midr = (fr + lr) / 2;
		if (target == matrix[midr][0]) return true;
		if (target > matrix[midr][0]) fr = midr + 1;
		if (target < matrix[midr][0]) lr = midr - 1;
	}
	midr = lr;
	while (fc <= lc) {
		midc = (fc + lc) / 2;
		if (target == matrix[midr][midc]) return true;
		if (target > matrix[midr][midc]) fc = midc + 1;
		if (target < matrix[midr][midc]) lc = midc - 1;
	}
	return false;
}
//No 75 Sort Colors
void solution::sortColors(vector<int> & nums)
{
	if (nums.empty()) return;
	int red = 0, blue = nums.size() - 1;
	for (int i = 0; i <= blue; ++i) {
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
	for (int i = 0; i < ans.size(); ++i) {
		ans[i].push_back(n);
	}
	ans_2 = combine(n - 1, k);
	for (int i = 0; i < ans_2.size(); ++i) {
		ans.push_back(ans_2[i]);
	}
	return ans;
}
//No 78 Subsets
vector<vector<int>> solution::subsets(vector<int> & nums)
{
	vector<vector<int>> ans;
	ans.push_back(vector<int>());
	sort(nums.begin(), nums.end());
	for (int i = 0; i < nums.size(); ++i) {
		int size = ans.size();
		for (int j = 0; j < size; ++j) {
			ans.push_back(ans[j]);
			ans.back().push_back(nums[i]);
		}
	}
	return ans;
}
//No 79 Word Search
bool solution::exist(vector<vector<char>> & board, string word)
{
	if (word.length() == 0) return true;
	if (board.size() == 0 || board[0].size() == 0) return false;
	int rows = board.size();
	int cols = board[0].size();
	vector<vector<bool>> enable(rows, vector<bool>(cols, 1));
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (board[i][j] == word[0] && exploreWordSearch(i, j, enable, 0, board, word)) return true;
		}
	}
	return false;
}


/* used in No 79 Word Search */
bool solution::exploreWordSearch(int row, int col, vector<vector<bool>> & enable, int position, const vector<vector<char>> & board, const string word)
{
	if (position == word.length() - 1) return true;
	enable[row][col] = 0;
	bool res = false;
	if (row > 0 && enable[row - 1][col] == 1 && board[row - 1][col] == word[position + 1]) {
		res = res || exploreWordSearch(row - 1, col, enable, position + 1, board, word);
	}
	if (row < board.size() - 1 && enable[row + 1][col] == 1 && board[row + 1][col] == word[position + 1]) {
		res = res || exploreWordSearch(row + 1, col, enable, position + 1, board, word);
	}
	if (col > 0 && enable[row][col - 1] == 1 && board[row][col - 1] == word[position + 1]) {
		res = res || exploreWordSearch(row, col - 1, enable, position + 1, board, word);
	}
	if (col < board[0].size() - 1 && enable[row][col + 1] == 1 && board[row][col + 1] == word[position + 1]) {
		res = res || exploreWordSearch(row, col + 1, enable, position + 1, board, word);
	}
	enable[row][col] = 1;
	return res;
}

//No 80 Remove Duplicates from Sorted Array II
int solution::removeDuplicates(vector<int> & nums)
{
	if (nums.size() <= 2) return nums.size();
	for (int i = 2; i < nums.size();) {
		if (nums[i] == nums[i - 1] && nums[i] == nums[i - 2]) {
			nums.erase(nums.begin() + i);
		}
		else ++i;
	}
	return nums.size();
}
//No 81 Search in Rotated Sorted Array II
bool solution::search(vector<int> & nums, int target)
{
	if (nums.size() == 0) return false;
	int start = 0, end = nums.size() - 1;
	while (start <= end) {
		int mid = (start + end) / 2;
		if (nums[mid] == target) return true;
		if (nums[mid] > nums[start]) {
			if (nums[start] <= target && target < nums[mid]) {
				end = mid - 1;
			}
			else {
				start = mid + 1;
			}
		}
		else if (nums[mid] < nums[start]) {
			if (nums[mid] < target && target < nums[start]) {
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
ListNode* solution::deleteDuplicates(ListNode * head)
{
	if (head == nullptr || head->next == nullptr) return head;
	ListNode * helper = new ListNode(0);
	helper->next = head;
	ListNode * pre = helper;
	ListNode * cur = head;
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
int solution::largestRectangleArea(vector<int> & heights)
{
	int res = 0;
	stack<int> st;
	heights.push_back(0);
	for (int i = 0; i < heights.size(); ++i) {
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
int solution::maximalRectangle(vector<vector<char>> & matrix)
{
	if (matrix.empty() || matrix[0].empty()) return 0;
	int res = 0;
	int m = matrix.size(), n = matrix[0].size();
	vector<int> height(n + 1, 0);
	for (int i = 0; i < m; ++i) {
		stack<int> st;
		for (int j = 0; j < n + 1; ++j) {
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
ListNode* solution::partition(ListNode * head, int x)
{
	ListNode* dummy = new ListNode(-1);
	dummy->next = head;
	ListNode * pre = dummy, *cur = head;
	while (pre->next && pre->next->val < x) pre = pre->next;
	cur = pre;
	while (cur->next) {
		if (cur->next->val < x) {
			ListNode* tmp = cur->next;
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
	for (int i = 0; i < pow(2, n); ++i) {
		ans.push_back((i / 2) ^ i);
	}
	return ans;
}
//No 90 Subsets II
vector<vector<int>> solution::subsetsWithDup(vector<int> & nums)
{
	vector<vector<int>> ans;
	ans.push_back(vector<int>());
	if (nums.empty()) return ans;
	sort(nums.begin(), nums.end());
	for (int i = 0; i < nums.size();) {
		int count = 1;
		while (i + count < nums.size() && nums[i] == nums[i + count])
			count++;
		int size = ans.size();
		for (int j = 0; j < size; ++j) {
			vector<int> tmp = ans[j];
			for (int k = 0; k < count; ++k) {
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
	if (s.empty() || s.size() > 1 && s[0] == '0') return 0;
	vector<int> dp(s.size() + 1, 0);
	dp[0] = 1;
	for (int i = 1; i < dp.size(); ++i) {
		dp[i] = (s[i - 1] == '0') ? 0 : dp[i - 1];
		if (i > 1 && (s[i - 2] == '1' || (s[i - 2] == '2' && s[i - 1] <= '6'))) {
			dp[i] += dp[i - 2];
		}
	}
	return dp.back();
}
//No 92 Reverse Linked List II
ListNode* solution::reverseBetween(ListNode * head, int m, int n)
{
	ListNode* dummy = new ListNode(-1);
	dummy->next = head;
	ListNode * cur = dummy, *pre, *front = nullptr, *last;
	for (int i = 0; i < m - 1; ++i) {
		cur = cur->next;
	}
	pre = cur;
	last = cur->next;
	for (int i = m; i <= n; ++i) {
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
void solution::restoreIpDfs(string s, vector<string> & ans, int dotNum, string partStr)
{
	if (dotNum > 4) return;
	if (dotNum == 4 && s.size() == 0) {
		ans.push_back(partStr);
		return;
	}
	if (dotNum <= 3) {
		for (int i = 1; i <= 3; ++i) {
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
	TreeNode* p = root;
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
	return generateTreesDfs(1, n + 1);
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
	for (int i = left; i < right; ++i) {
		vector<TreeNode*> leftSubTree = generateTreesDfs(left, i);
		vector<TreeNode*> rightSubTree = generateTreesDfs(i + 1, right);
		for (int j = 0; j < leftSubTree.size(); ++j) {
			for (int k = 0; k < rightSubTree.size(); ++k) {
				TreeNode* root = new TreeNode(i);
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
	for (int i = 2; i < n + 1; ++i) {
		for (int j = 0; j < i; ++j) {
			dp[i] += dp[j] * dp[i - j - 1];
		}
	}
	return dp[n];
}
//No 98 Validate Binary Search Tree
bool solution::isValidBST(TreeNode * root)
{
	if (root == NULL) return true;
	vector<int> vals;
	inorderBinaryTree(root, vals);
	for (int i = 1; i < vals.size(); ++i) {
		if (vals[i - 1] >= vals[i]) return false;
	}
	return true;
}

/*used in 98 Validate Binary Search Tree*/
void solution::inorderBinaryTree(TreeNode * root, vector<int> & vals)
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
		for (int i = 0; i < size; ++i) {
			TreeNode* node = q.front();
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
			TreeNode* cur = s1.top();
			s1.pop();
			level.push_back(cur->val);
			if (cur->left) s2.push(cur->left);
			if (cur->right) s2.push(cur->right);
		}
		if (!level.empty()) ans.push_back(level);
		level.clear();
		while (!s2.empty()) {
			TreeNode* cur = s2.top();
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
TreeNode* solution::buildTree_preIn(vector<int> & preorder, vector<int> & inorder)
{
	return buildTreeRes_preIn(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
}

TreeNode* solution::buildTreeRes_preIn(vector<int> & preorder, int preStart, int preEnd, vector<int> & inorder, int inStart, int inEnd)
{
	if (preStart > preEnd || inStart > inEnd) return NULL;
	TreeNode * root = new TreeNode(preorder[preStart]);
	int rootIndex = inStart;
	for (rootIndex = inStart; rootIndex <= inEnd; ++rootIndex) {
		if (inorder[rootIndex] == root->val) break;
	}
	int leftNum = rootIndex - inStart;
	root->left = buildTreeRes_preIn(preorder, preStart + 1, preStart + leftNum, inorder, inStart, rootIndex - 1);
	root->right = buildTreeRes_preIn(preorder, preStart + leftNum + 1, preEnd, inorder, rootIndex + 1, inEnd);
	return root;
}

//No 106 Construct Binary Tree from Inorder and Postorder Traversal
TreeNode * solution::buildTree_inPos(vector<int> & inorder, vector<int> & postorder)
{
	return buildTreeRes_inPos(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
}

TreeNode* solution::buildTreeRes_inPos(vector<int> & inorder, int inStart, int inEnd, vector<int> & postorder, int posStart, int posEnd)
{
	if (inStart > inEnd || posStart > posEnd) return NULL;
	TreeNode * root = new TreeNode(postorder[posEnd]);
	int rootIndex = inStart;
	for (rootIndex = inStart; rootIndex <= inEnd; ++rootIndex) {
		if (inorder[rootIndex] == root->val) break;
	}
	int leftNum = rootIndex - inStart;
	root->left = buildTreeRes_inPos(inorder, inStart, rootIndex - 1, postorder, posStart, posStart + leftNum - 1);
	root->right = buildTreeRes_inPos(inorder, rootIndex + 1, inEnd, postorder, posStart + leftNum, posEnd - 1);
	return root;
}



//No 109 Convert Sorted List to Binary Search Tree
TreeNode * solution::sortedListToBST(ListNode * head)
{
	//int len = 0;
	//ListNode* p = head;
	//while (p) {
	//	len++;
	//	p = p->next;
	//}
	//return buildBST(head, 0, len - 1);
	if (!head) return nullptr;
	else if (!head->next) return new TreeNode(head->val);
	ListNode* pre = head, * p = pre->next, * q = p->next;
	while (q && q->next) {
		pre = pre->next;
		p = pre->next;
		q = q->next->next;
	}
	pre->next = nullptr;
	TreeNode* root = new TreeNode(p->val);
	root->left = sortedListToBST(head);
	root->right = sortedListToBST(p->next);
	return root;
}

TreeNode* solution::buildBST(ListNode * &list, int start, int end)
{
	if (start > end) return NULL;
	int mid = (start + end) / 2;
	TreeNode * left = buildBST(list, start, mid - 1);
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

void solution::pathSumDfs(vector<vector<int>> &ans, vector<int> &tmp, TreeNode * root, int sum)
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
	if (root->left != NULL) {
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
int solution::minimumTotal(vector<vector<int>> & triangle)
{
	int rows = triangle.size();
	vector<int> dp(triangle.back());
	for (int i = rows - 2; i >= 0; --i) {
		for (int j = 0; j <= i; ++j) {
			dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j];
		}
	}
	return dp[0];
}

//No 127 Word Ladder
int solution::ladderLength(string beginWord, string endWord, vector<string> & wordList)
{
	unordered_set<string> dict(wordList.begin(), wordList.end());
	if (dict.find(endWord) == dict.end()) return 0;
	int ans = 0;
	unordered_set<string> head;
	unordered_set<string> tail;
	head.insert(beginWord); tail.insert(endWord);
	while (!head.empty() || !tail.empty()) {
		if (head.size() < tail.size()) swap(head, tail);
		ans++;
		unordered_set<string> tmp;
		for (string w : head) {
			for (int i = 0; i < endWord.size(); ++i) {
				char ctmp = w[i];
				for (char j = 'a'; j <= 'z'; ++j) {
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
	if (!root->left && !root->right) return curSum * 10 + valRoot;
	return sumNumbersRes(root->left, curSum * 10 + valRoot) + sumNumbersRes(root->right, curSum * 10 + valRoot);
}

//No 130 Surrounded Regions
void solution::solveSurronded(vector<vector<char>> & board)
{
	for (int i = 0; i < board.size(); ++i) {
		for (int j = 0; j < board[i].size(); ++j) {
			if ((i == 0 || i == board.size() - 1 || j == 0 || j == board[i].size() - 1) && board[i][j] == 'O')
				solveSurrondedDfs(board, i, j);
		}
	}

	for (int i = 0; i < board.size(); ++i) {
		for (int j = 0; j < board[i].size(); ++j) {
			if (board[i][j] == 'O') board[i][j] = 'X';
			if (board[i][j] == 'A') board[i][j] = 'O';
		}
	}
}
void solution::solveSurrondedDfs(vector<vector<char>> & board, int i, int j)
{
	if (board[i][j] == 'O') {
		board[i][j] = 'A';
		if (i > 0 && board[i - 1][j] == 'O') solveSurrondedDfs(board, i - 1, j);
		if (i < board.size() - 1 && board[i + 1][j] == 'O') solveSurrondedDfs(board, i + 1, j);
		if (j > 0 && board[i][j - 1] == 'O') solveSurrondedDfs(board, i, j - 1);
		if (j < board[i].size() - 1 && board[i][j + 1] == 'O') solveSurrondedDfs(board, i, j + 1);
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
void solution::partitionDfs(vector<vector<string>> & ans, vector<string> out, string s, int start)
{
	if (start == s.size()) {
		ans.push_back(out);
		return;
	}
	for (int i = start; i < s.length(); ++i) {
		if (isValidPali(s, start, i)) {
			out.push_back(s.substr(start, i - start + 1));
			partitionDfs(ans, out, s, i + 1);
			out.pop_back();
		}
	}
}

//No 133 Clone Graph
UndirectedGraphNode* solution::cloneGraph(UndirectedGraphNode * node)
{
	unordered_map<int, UndirectedGraphNode*> umap;
	return cloneGraphHelper(node, umap);
}

UndirectedGraphNode* solution::cloneGraphHelper(UndirectedGraphNode * node, unordered_map<int, UndirectedGraphNode*> & umap)
{
	if (!node) return node;
	if (umap.count(node->label)) return umap[node->label];
	UndirectedGraphNode* newNode = new UndirectedGraphNode(node->label);
	umap[node->label] = newNode;
	for (int i = 0; i < node->neighbors.size(); ++i) {
		newNode->neighbors.push_back(cloneGraphHelper(node->neighbors[i], umap));
	}
	return newNode;
}

//No 134 Gas Station
int solution::canCompleteCircuit(vector<int> & gas, vector<int> & cost)
{
	int start = 0, sum = 0, total = 0;
	for (int i = 0; i < gas.size(); ++i) {
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
int solution::singleNumber(vector<int> & nums)
{
	/* sort */
	//sort(nums.begin(), nums.end());
	//if (nums[0] != nums[1]) return nums[0];
	//for (int i = 1; i < nums.size() - 1; ++i) {
	//	if (nums[i] != nums[i - 1] && nums[i] != nums[i + 1])
	//		return nums[i];
	//}
	//return nums.back();

	/* calculate per bit */
	//int len = nums.size(), ans = 0;
	//for (int i = 0; i < 32; ++i) {
	//	int count = 0;
	//	int mask = 1 << i;
	//	for (int j = 0; j < len; ++j) {
	//		if (nums[j] & mask)count++;
	//	}
	//	if (count % 3) ans |= mask;
	//}
	//return ans;

	/* 2 var xor*/
	int a = 0, b = 0;
	for (auto n : nums) {
		b = (b ^ n) & (~a);
		a = (a ^ n) & (~b);
	}
	return b;
}
//No 138 Copy List with Random Pointer
RandomListNode* solution::copyRandomList(RandomListNode * head)
{
	if (!head) return NULL;
	RandomListNode* ans = new RandomListNode(head->label);
	RandomListNode* node = ans;
	RandomListNode* cur = head->next;
	map<RandomListNode*, RandomListNode*> m;
	m[head] = ans;
	while (cur) {
		RandomListNode* tmp = new RandomListNode(cur->label);
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
bool solution::wordBreak(string s, vector<string> & wordDict)
{
	vector<bool> flags(s.length() + 1, false);
	unordered_set<string> dict(wordDict.begin(), wordDict.end());
	flags[0] = true;
	for (int i = 0; i < s.length() + 1; ++i) {
		for (int j = i - 1; j >= 0; --j) {
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
	if (!head || !head->next) return false;
	ListNode * fast = head, *slow = head;
	while (fast && fast->next) {
		slow = slow->next;
		fast = fast->next->next;
		if (fast == slow) return true;
	}
	return false;
}

//No 142 Linked List Cycle II
ListNode* solution::detectCycle(ListNode * head)
{
	if (!head || !head->next) return NULL;
	ListNode * fast = head, *slow = head;
	while (fast && fast->next) {
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
	while (fast && fast->next) {
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
	ListNode* first = head;
	mid = pre;
	while (first && mid) {
		ListNode* ftmp = first->next;
		ListNode* mtmp = mid->next;
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
	stack<TreeNode*> s;
	s.push(root);
	while (!s.empty()) {
		TreeNode* t = s.top();
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
	stack<TreeNode*> s;
	deque<int> dq;
	s.push(root);
	while (!s.empty()) {
		TreeNode* t = s.top();
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
ListNode* solution::insertionSortList(ListNode * head)
{
	if (!head || !head->next) return head;
	ListNode * dummy = new ListNode(0);
	ListNode * cur = head;
	ListNode * pre = dummy;
	while (cur) {
		ListNode* next = cur->next;
		pre = dummy;
		while (pre->next && pre->next->val <= cur->val) {
			pre = pre->next;
		}
		cur->next = pre->next;
		pre->next = cur;
		cur = next;
	}
	return dummy->next;
}

//No 148 Sort List
ListNode* solution::sortList(ListNode * head)
{
	if (!head || !head->next) return head;
	ListNode * fast = head, *slow = head, *pre = head;
	while (fast && fast->next) {
		pre = slow;
		fast = fast->next->next;
		slow = slow->next;
	}
	pre->next = NULL;
	return mergeSortedList(sortList(head), sortList(slow));
}

ListNode* solution::mergeSortedList(ListNode * list1, ListNode * list2)
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
int solution::evalRPN(vector<string> & tokens)
{
	if (tokens.size() == 1) {
		return atoi(tokens[0].c_str());
	}
	stack<int> s;
	for (int i = 0; i < tokens.size(); ++i) {
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
int solution::maxProduct(vector<int> & nums)
{
	long long maxPro = nums[0], preMax = nums[0], preMin = nums[0];
	long long curMax, curMin;
	if (nums.size() == 1) return maxPro > INT32_MAX ? INT32_MAX : (int)maxPro;
	for (int i = 1; i < nums.size(); ++i) {
		curMin = min(min(preMax * nums[i], preMin * nums[i]), (long long)nums[i]);
		curMax = max(max(preMax * nums[i], preMin * nums[i]), (long long)nums[i]);
		maxPro = max(curMax, maxPro);
		preMax = curMax;
		preMin = curMin;
	}
	return maxPro > INT32_MAX ? INT32_MAX : (int)maxPro;
}

//No 153 Find Minimum in Rotated Sorted Array
int solution::findMin(vector<int> & nums)
{
	if (nums.size() == 1) return nums[0];
	for (int i = 1; i < nums.size(); ++i) {
		if (nums[i] < nums[i - 1]) return nums[i];
	}
	return nums[0];
}

//No 160 Intersection of Two Linked Lists
ListNode* solution::getIntersectionNode(ListNode * headA, ListNode * headB)
{
	if (!headA || !headB) return NULL;
	int lenA = 0, lenB = 0;
	ListNode * helpA = headA, *helpB = headB;
	while (helpA) {
		helpA = helpA->next;
		lenA++;
	}
	while (helpB) {
		helpB = helpB->next;
		lenB++;
	}
	if (lenA > lenB) {
		for (int i = 0; i < lenA - lenB; ++i) {
			headA = headA->next;
		}
	}
	else {
		for (int i = 0; i < lenB - lenA; ++i) {
			headB = headB->next;
		}
	}
	while (headA && headB && headA != headB) {
		headA = headA->next;
		headB = headB->next;
	}
	return(headA && headB) ? headA : NULL;
}

//No 162 Find Peak Element
int solution::findPeakElement(vector<int> & nums)
{
	if (nums.size() == 1) return 0;
	if (nums[0] > nums[1]) return 0;
	for (int i = 1; i < nums.size() - 1; ++i) {
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
	long long int r = n % d;
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
		r = r % d;
	}
	return result;
}

//No 167 Two Sum II - Input array is sorted
vector<int> solution::twoSum(vector<int> & numbers, int target)
{
	vector<int> ans;
	int left = 0, right = numbers.size() - 1;
	int sum;
	while (left < right) {
		sum = numbers[left] + numbers[right];
		if (sum == target) {
			ans.push_back(left + 1);
			ans.push_back(right + 1);
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
int solution::majorityElement(vector<int> & nums)
{
	int elem = 0, count = 0;
	for (int i = 0; i < nums.size(); ++i) {
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
	for (int i = 0; i < s.length(); ++i) {
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
void solution::rotate(vector<int> & nums, int k)
{
	int len = nums.size();
	k = k % len;
	for (int i = 0; i < len - k; ++i) {
		nums.push_back(nums.front());
		nums.erase(nums.begin());
	}
	return;
}

//No 190 Reverse Bits
uint32_t solution::reverseBits(uint32_t n)
{
	uint32_t ans = 0;
	int count = 0;
	while (count < 32) {
		ans = ans << 1;
		if (n & 1) ans++;
		n = n >> 1;
		count++;
	}
	return ans;
	//another solution inplace
	//for (int i = 0; i <= 15; ++i) {
	//	int flag1 = 1 << i, flag2 = 1 << (31 - i);
	//	if (((n & flag1) >> i) != ((n & flag2) >> (31 - i))) {
	//		n = n ^ flag1;
	//		n = n ^ flag2;
	//	}
	//}
	//return n;
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
int solution::rob(vector<int> & nums)
{
	if (nums.empty()) return 0;
	if (nums.size() == 1) return nums[0];
	int a = 0, b = 0;
	for (int i = 0; i < nums.size(); ++i) {
		if (i % 2 == 0) a = max(a + nums[i], b);
		else b = max(b + nums[i], a);
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

void solution::rightSVdfs(TreeNode * root, int deep, vector<int> & ans)
{
	if (!root) return;
	if (deep >= ans.size()) ans.push_back(root->val);
	rightSVdfs(root->right, deep + 1, ans);
	rightSVdfs(root->left, deep + 1, ans);
}


//No 200 Number of Islands
int solution::numIslands(vector<vector<char>> & grid)
{
	if (grid.empty()) return 0;
	int num = 0;
	for (int i = 0; i < grid.size(); ++i) {
		for (int j = 0; j < grid[0].size(); ++j) {
			if (grid[i][j] == '1') {
				DfsNumIslands(grid, i, j);
				num++;
			}
		}
	}
	return num;
}

void solution::DfsNumIslands(vector<vector<char>> & grid, int row, int col)
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
	//int count = 0;
	//while (n != m) {
	//	n >>= 1;
	//	m >>= 1;
	//	count++;
	//}
	//return (m << count);
	while (m < n) {
		n = n & (n - 1);
	}
	return n;
}

//No 202 Happy Number
bool solution::isHappy(int n)
{
	set<int> squSumSet;
	while (n != 1) {
		int t = 0;
		while (n) {
			t = t + (n % 10) * (n % 10);
			n = n / 10;
		}
		n = t;
		if (squSumSet.count(t) != 0) break;
		else squSumSet.insert(t);
	}
	return n == 1;
}

//No 203 Remove Linked List Elements
ListNode* solution::removeElements(ListNode * head, int val)
{
	if (head == nullptr) return head;
	ListNode * helper = new ListNode(0);
	helper->next = head;
	ListNode * p = helper;
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
	vector<bool> notPrimes(n, false);
	int count = 1;
	for (int x = 3; x * x < n; x = x + 2) {
		if (notPrimes[x]) {
			continue;
		}
		else {
			for (int j = x * x; j < n; j = j + 2 * x) {
				notPrimes[j] = true;
			}
		}
	}
	for (int x = 3; x < n; x = x + 2) {
		if (!notPrimes[x]) count++;
	}
	return count;
}

bool solution::isPrime(int x, vector<int> primes)
{
	for (int i = 0; primes[i] * primes[i] <= x; ++i) {
		if (x % primes[i] == 0) return false;
	}
	return true;
}


//No 205 Isomorphic Strings
bool solution::isIsomorphic(string s, string t)
{
	if (s.size() != t.size()) return false;
	unordered_map<char, vector<int>> hashS;
	unordered_map<char, vector<int>> hashT;
	for (int i = 0; i < s.size(); ++i) {
		hashS[s[i]].push_back(i);
		hashT[t[i]].push_back(i);
	}
	for (int i = 0; i < s.size(); ++i) {
		if (hashS[s[i]].size() == 1 && hashT[t[i]].size() == 1) continue;
		if (hashS[s[i]] != hashT[t[i]]) return false;
	}
	return true;
}

//No 206 Reverse Linked List
ListNode* solution::reverseList(ListNode * head)
{
	if (!head || !head->next) return head;
	ListNode * p = head;
	ListNode * q = head;
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
bool solution::canFinish(int numCourses, vector<pair<int, int>> & prerequisites)
{
	vector<int> heads(numCourses, -1);
	vector<int> enDegree(numCourses, 0);
	vector<int> points, args;
	pair<int, int> p;
	int from, to, count = 0, len = prerequisites.size();
	for (int i = 0; i < len; ++i) {
		p = prerequisites[i];
		from = p.second;
		to = p.first;
		++enDegree[to];
		args.push_back(heads[from]);
		points.push_back(to);
		heads[from] = count++;
	}
	queue<int> q;
	for (int i = 0; i < numCourses; ++i) {
		if (enDegree[i] == 0) q.push(i);
	}
	while (!q.empty()) {
		from = q.front();
		q.pop();
		to = heads[from];
		while (to != -1) {
			if (--enDegree[points[to]] == 0) q.push(points[to]);
			to = args[to];
		}
	}
	for (int i = 0; i < numCourses; ++i) {
		if (enDegree[i] > 0) return false;
	}
	return true;
}
bool solution::canFinish(int numCourses, vector<vector<int>>& prerequisites)
{
	vector<int>ind(numCourses, 0);
	unordered_map<int, vector<int>>m;
	for (auto p : prerequisites) {
		ind[p[0]]++;
		m[p[1]].push_back(p[0]);
	}
	queue<int>q;
	for (int i = 0; i < numCourses; ++i) {
		if (ind[i] == 0) q.push(i);
	}
	int count = 0;
	while (!q.empty()) {
		int cur = q.front();
		q.pop();
		++count;
		for (auto next : m[cur]) {
			--ind[next];
			if (ind[next] == 0) q.push(next);
		}
	}
	return count == numCourses;
}

//No 209 Minimum Size Subarray Sum
int solution::minSubArrayLen(int s, vector<int> & nums)
{
	if (nums.empty()) return 0;
	int left = 0, right = 0, sum = 0, res = nums.size() + 1;
	while (right < nums.size()) {
		while (sum < s && right < nums.size()) {
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
vector<int> solution::findOrder(int numCourses, vector<pair<int, int>> & prerequisites)
{
	vector<int> heads(numCourses, -1);
	vector<int> enDegree(numCourses, 0);
	vector<int> points, args, ans;
	pair<int, int> p;
	int from, to, count = 0, len = prerequisites.size();
	for (int i = 0; i < len; ++i) {
		p = prerequisites[i];
		from = p.second;
		to = p.first;
		++enDegree[to];
		args.push_back(heads[from]);
		points.push_back(to);
		heads[from] = count++;
	}
	queue<int> q;
	for (int i = 0; i < numCourses; ++i) {
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
	for (int i = 0; i < numCourses; ++i) {
		if (enDegree[i] > 0) return vector<int>();
	}
	return ans;
	//simply solution bfs topological order
	/*unordered_map<int, vector<int>>pre;
	vector<int>in(numCourses, 0);
	vector<int>ans;
	for (auto p : prerequisites) {
		pre[p[1]].push_back(p[0]);
		in[p[0]]++;
	}
	queue<int>q;
	for (int i = 0; i < numCourses; ++i) {
		if (in[i] == 0)q.push(i);
	}
	while (!q.empty()) {
		int cur = q.front();
		q.pop();
		in[cur] = -1;
		for (auto n : pre[cur]) {
			in[n]--;
			if (in[n] == 0) q.push(n);
		}
		ans.push_back(cur);
	}
	return ans.size() == numCourses ? ans : vector<int>();*/
}

//No 213 House Robber II
int solution::rob2(vector<int> & nums)
{
	if (nums.size() == 0) return 0;
	if (nums.size() == 1) return nums[0];
	vector<int> tmp1(nums.begin(), nums.end() - 1);
	vector<int> tmp2(nums.begin() + 1, nums.end());
	return max(robPlan(tmp1), robPlan(tmp2));
}

int solution::robPlan(vector<int> & nums)
{
	vector<int> dp(nums.size(), 0);
	dp[0] = nums[0];
	dp[1] = max(nums[0], nums[1]);
	dp[2] = max(nums[1], nums[0] + nums[2]);
	for (int i = 3; i < nums.size(); ++i) {
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
	for (int q = 1; q < t.length(); ++q) {
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
	int i = 0, j = 0;
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
	string revS = s;
	reverse(revS.begin(), revS.end());
	string t = s + '#' + revS;
	vector<int>next(t.size(), -1);
	int k = -1;
	//calculate vector next
	for (int q = 1; q < t.length(); ++q) {
		while (k > -1 && t[k + 1] != t[q]) {
			k = next[k];
		}
		if (t[k + 1] == t[q]) {
			k++;
		}
		next[q] = k;
	}
	cout << next[t.size() - 1] << endl;
	string ans = revS.substr(0, revS.size() - next[t.size() - 1] - 1) + s;
	return ans;
}

//No 215 Kth Largest Element in an Array
int solution::findKthLargest(vector<int> & nums, int k)
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

void solution::combinationSum3Dfs(int k, int n, int level, vector<int> & out, vector<vector<int>> & ans)//&
{
	if (n < 0 || out.size()>k) return;
	if (n == 0 && out.size() == k) ans.push_back(out);
	for (int i = level; i <= 9; ++i) {
		out.push_back(i);
		combinationSum3Dfs(k, n - i, i + 1, out, ans);
		out.pop_back();
	}
}



//No 217 Contains Duplicate
bool solution::containsDuplicate(vector<int> & nums)
{
	set<int>numSet;
	for (int i = 0; i < nums.size(); ++i) {
		if (numSet.find(nums[i]) != numSet.end()) return true;
		else {
			numSet.insert(nums[i]);
		}
	}
	return false;
}

//No 219 Contains Duplicate II
bool solution::containsNearbyDuplicate(vector<int> & nums, int k)
{
	map<int, int>numMap;
	for (int i = 0; i < nums.size(); ++i) {
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
bool solution::containsNearbyAlmostDuplicate(vector<int> & nums, int k, int t)
{
	set<long> numSet;
	long lt = t;
	for (int i = 0; i < nums.size(); ++i) {
		if (i > k) numSet.erase(nums[i - k - 1]);
		set<long>::iterator it = numSet.lower_bound(nums[i] - lt);
		if (it != numSet.end() && (*it) - nums[i] <= lt) return true;
		numSet.insert(nums[i]);
	}
	return false;
}

//No 221 Maximal Square
int solution::maximalSquare(vector<vector<char>> & matrix)//must be square
{
	if (matrix.empty()) return 0;
	int ans = 0;
	vector<vector<int>> dp(matrix.size(), vector<int>(matrix[0].size(), 0));
	for (int i = 0; i < matrix.size(); ++i) {
		for (int j = 0; j < matrix[0].size(); ++j) {
			if (i > 0 && j > 0 && matrix[i][j] == '1') {
				dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
			}
			else {
				dp[i][j] = matrix[i][j] - '0';
			}
			ans = max(ans, dp[i][j]);
		}
	}
	return ans * ans;
}

//No 222 Count Complete Tree Nodes
int solution::countNodes(TreeNode * root)
{
	if (root == nullptr) return 0;
	int hLeft = 0, hRight = 0;
	TreeNode * pLeft = root, *pRight = root;
	while (pLeft) {
		hLeft++;
		pLeft = pLeft->left;
	}
	while (pLeft) {
		hRight++;
		pRight = pRight->right;
	}
	if (hLeft == hRight) return pow(2, hLeft) - 1;
	return 1 + countNodes(root->left) + countNodes(root->right);

	//if (!root) return 0;
	//int hl = 0, hr = 0;
	//TreeNode *l = root, *r = root;
	//while (l) { hl++;l = l->left; }
	//while (r) { hr++;r = r->right; }
	//if (hl == hr) return pow(2, hl) - 1;
	//return 1 + countNodes(root->left) + countNodes(root->right);
}

//No 223 Rectangle Area
int solution::computeArea(int A, int B, int C, int D, int E, int F, int G, int H)
{
	long crossLeft = max(A, E);
	long crossRight = min(C, G);
	long crossUp = min(D, H);
	long crossBottom = max(B, F);
	int overLap = max(0L, crossRight - crossLeft) * max(0L, crossUp - crossBottom);
	return (C - A) * (D - B) + (G - E) * (H - F) - overLap;
}

//No 226 Invert Binary Tree
TreeNode * solution::invertTree(TreeNode * root)
{
	if (root == nullptr) return nullptr;
	TreeNode * tmp = root->left;
	root->left = root->right;
	root->right = tmp;
	invertTree(root->left);
	invertTree(root->right);
	return root;
}

//No 227 Basic Calculator II
int solution::calculate(string s)
{
	int ans = 0, temp = 0;
	char op = '+';
	stack<int> st;
	for (int i = 0; i < s.size(); ++i) {
		if (s[i] >= '0' && s[i] <= '9') {
			temp = temp * 10 + (s[i] - '0');
		}
		if ((s[i]<'0' || s[i]>'9') && s[i] != ' ' || i == s.size() - 1) {
			if (op == '+') {
				st.push(temp);
			}
			else if (op == '-') {
				st.push(-temp);
			}
			else if (op == '*') {
				int num = st.top();
				st.pop();
				st.push(num * temp);
			}
			else if (op == '/') {
				int num = st.top();
				st.pop();
				st.push(num / temp);
			}
			op = s[i];
			temp = 0;
		}
	}
	while (!st.empty()) {
		ans += st.top();
		st.pop();
	}
	return ans;
}

//No 228 Summary Ranges
vector<string> solution::summaryRanges(vector<int> & nums)
{
	vector<string> ans;
	string out;
	if (nums.empty()) return ans;
	int index = 0, range = 1;
	while (index < nums.size()) {
		range = 1;
		while (index + range < nums.size() && nums[index + range] - nums[index] == range) ++range;
		if (range <= 1) out = to_string(nums[index]);
		else out = to_string(nums[index]) + "->" + to_string(nums[index + range - 1]);
		ans.push_back(out);
		index += range;
	}
	return ans;
}

//No 229 Majority Element II
vector<int> solution::majorityElement2(vector<int> & nums)
{
	/*used map, not sure if it runs in O(1) space*/
	/*vector<int> ans;
	map<int, int>countMap;
	int th = nums.size() / 3;
	for (int i = 0; i < nums.size(); ++i) {
		if (countMap.find(nums[i]) != countMap.end()) {
			countMap[nums[i]]++;
		}
		else {
			countMap[nums[i]] = 1;
		}
	}
	for (map<int, int>::iterator it = countMap.begin(); it != countMap.end(); ++it) {
		if (it->second > th) ans.push_back(it->first);
	}
	return ans;*/
	/********************************************/
	vector<int>res;
	int a = 0, b = 0, cnt1 = 0, cnt2 = 0, n = nums.size();
	for (int num : nums) {
		if (num == a)++cnt1;
		else if (num == b)++cnt2;
		else if (cnt1 == 0) {
			a = num;
			++cnt1;
		}
		else if (cnt2 == 0) {
			b = num;
			++cnt2;
		}
		else {
			--cnt1; 
			--cnt2;
		}
	}
	cnt1 = cnt2 = 0;
	for (int num : nums) {
		if (num == a) ++cnt1;
		else if (num == b) ++cnt2;
	}
	if (cnt1 > n / 3) res.push_back(a);
	if (cnt2 > n / 3) res.push_back(b);
	return res;
}

//No 230 Kth Smallest Element in a BST
int solution::kthSmallest(TreeNode * root, int k)
{
	vector<int> val;
	inorderBinaryTree(root, val);//The in-order traversal of the binary search tree is in ascending order
	//defined in line 735
	return val[k - 1];
}


//No 231 Power of Two
bool solution::isPowerOfTwo(int n)
{
	if (n <= 0) return false;
	while (n > 1) {
		if (n % 2 == 1)return false;
		n /= 2;
	}
	return true;
}

//No 234 Palindrome Linked List
bool solution::isPalindrome(ListNode * head)
{
	if (head == nullptr || head->next == nullptr) return true;
	ListNode * fast = head, *slow = head;
	while (fast->next && fast->next->next) {
		fast = fast->next->next;
		slow = slow->next;
	}
	ListNode* last = slow->next, * pre = head;
	while (last->next) {
		ListNode* tmp = last->next;
		last->next = tmp->next;
		tmp->next = slow->next;
		slow->next = tmp;
	}
	while (slow->next) {
		slow = slow->next;
		if (pre->val != slow->val) return false;
		pre = pre->next;
	}
	return true;
}

//No 235 Lowest Common Ancestor of a Binary Search Tree
TreeNode* solution::lowestCommonAncestor(TreeNode * root, TreeNode * p, TreeNode * q)
{
	if (root->val > p->val && root->val > q->val) return lowestCommonAncestor(root->left, p, q);
	if (root->val < p->val && root->val < q->val) return lowestCommonAncestor(root->right, p, q);
	return root;
}

//No 236 Lowest Common Ancestor of a Binary Tree
TreeNode* solution::lowestCommonAncestor2(TreeNode * root, TreeNode * p, TreeNode * q)
{
	if (root == nullptr || root == p || root == q) return root;
	TreeNode * left = lowestCommonAncestor2(root->left, p, q);
	TreeNode * right = lowestCommonAncestor2(root->right, p, q);
	if (left != nullptr && right != nullptr) return root;
	return left == nullptr ? right : left;
}

//No 237 Delete Node in a Linked List
void solution::deleteNode(ListNode * node)
{
	node->val = node->next->val;
	node->next = node->next->next;
}

//No 238 Product of Array Except Self
vector<int> solution::productExceptSelf(vector<int> & nums)
{
	int len = nums.size();
	vector<int> ans(len, 1);
	int fromBegin = 1, fromLast = 1;
	for (int i = 0; i < nums.size(); ++i) {
		ans[i] *= fromBegin;
		fromBegin *= nums[i];
		ans[len - 1 - i] *= fromLast;
		fromLast *= nums[len - 1 - i];
	}
	return ans;
}

//No 240 Search a 2D Matrix II
bool solution::searchMatrix2(vector<vector<int>> & matrix, int target)
{
	//binary search
	if (matrix.empty() || matrix[0].empty()) return false;
	int rows = matrix.size(), cols = matrix[0].size();
	int rup = 0, rdown = rows - 1, cleft = 0, cright = cols - 1;
	int rmid, cmid;
	while (rup <= rdown) {
		rmid = (rup + rdown) / 2;
		if (matrix[rmid][0] == target) return true;
		else if (matrix[rmid][0] < target) rup = rmid + 1;
		else rdown = rmid - 1;
	}
	for (int i = 0; i <= rmid; ++i) {
		cleft = 0, cright = cols - 1;
		while (cleft <= cright) {
			cmid = (cleft + cright) / 2;
			if (matrix[i][cmid] == target) return true;
			else if (matrix[i][cmid] < target) cleft = cmid + 1;
			else cright = cmid - 1;
		}
	}
	return false;
}

//No 241 Different Ways to Add Parentheses
vector<int> solution::diffWaysToCompute(string input)
{
	vector<int> ans;
	for (int i = 0; i < input.size(); ++i) {
		char c = input[i];
		if (c > '9' || c < '0') {
			for (int a : diffWaysToCompute(input.substr(0, i))) {
				for (int b : diffWaysToCompute(input.substr(i + 1)))
					switch (c)
					{
					case '+':
						ans.push_back(a + b);
						break;
					case '-':
						ans.push_back(a - b);
						break;
					case '*':
						ans.push_back(a * b);
						break;
					default:
						break;
					}
			}
		}
	}
	if (ans.empty()) ans.push_back(stoi(input));
	return ans;
}

//No 242 Valid Anagram
bool solution::isAnagram(string s, string t)
{
	if (s.length() != t.length()) return false;
	unordered_map<char, int> letterHash;
	for (char c : s) {
		if (letterHash.find(c) == letterHash.end()) {
			letterHash.insert(pair<char, int>(c, 1));
		}
		else {
			letterHash.find(c)->second++;
		}
	}
	for (char c : t) {
		if (letterHash.find(c) == letterHash.end() || letterHash.find(c)->second <= 0) {
			return false;
		}
		else {
			letterHash.find(c)->second--;
		}
	}
	for (auto letter : letterHash) {
		if (letter.second > 0) return false;
	}
	return true;
}

//No 257 Binary Tree Paths
vector<string> solution::binaryTreePaths(TreeNode * root)
{
	/*recursive solution*/
	vector<string>ans;
	if (root == nullptr) return ans;
	string out;
	serchBtPaths(root, ans, out);
	return ans;
	/*****************/
}
/*recursive solution*/
void solution::serchBtPaths(TreeNode * root, vector<string> & ans, string out)
{
	if (root->left == nullptr && root->right == nullptr) {
		out = out + to_string(root->val);
		ans.push_back(out);
	}
	if (root->left != nullptr) serchBtPaths(root->left, ans, out + to_string(root->val) + "->");
	if (root->right != nullptr) serchBtPaths(root->right, ans, out + to_string(root->val) + "->");
}
/*****************/

//No 258 Add Digits
int solution::addDigits(int num)
{
	if (num <= 9) return num;
	int dSum = 0;
	while (num) {
		dSum += num % 10;
		num = num / 10;
	}
	return addDigits(dSum);
}

//No 260 Single Number III
vector<int> solution::singleNumber3(vector<int> & nums)
{
	map<int, int> m;
	map<int, int>::iterator mit;
	vector<int> ans;
	for (int i = 0; i < nums.size(); ++i) {
		mit = m.find(nums[i]);
		if (mit == m.end()) {
			m.insert(pair<int, int>(nums[i], 1));
		}
		else {
			mit->second++;
		}
	}
	for (mit = m.begin(); mit != m.end(); ++mit) {
		if (mit->second == 1) ans.push_back(mit->first);
	}
	return ans;
}

//No 263 Ugly Number
bool solution::isUgly(int num)
{
	if (num <= 0) return false;
	if (num == 1) return true;
	while (num > 1) {
		if (num % 2 != 0 && num % 3 != 0 && num % 5 != 0) return false;
		else {
			if (num % 2 == 0) num /= 2;
			if (num % 3 == 0) num /= 3;
			if (num % 5 == 0) num /= 5;
		}
	}
	return true;
}

//No 264 Ugly Number II
int solution::nthUglyNumber(int n)
{
	if (n <= 1) return 1;
	int t2 = 0, t3 = 0, t5 = 0;
	vector<int> dp(n);
	dp[0] = 1;
	for (int i = 1; i < n; ++i) {
		dp[i] = min(dp[t2] * 2, min(dp[t3] * 3, dp[t5] * 5));
		if (dp[i] == dp[t2] * 2) t2++;
		if (dp[i] == dp[t3] * 3) t3++;
		if (dp[i] == dp[t5] * 5) t5++;
	}
	return dp[n - 1];
}

//No 268 Missing Number
int solution::missingNumber(vector<int> & nums)
{
	int len = nums.size();
	long long sumAll = len * (len + 1) / 2;
	long long sum = 0;
	for (int i = 0; i < len; ++i) {
		sum += nums[i];
	}
	return sumAll - sum;
}

//No 306 Additive Number
bool solution::isAdditiveNumber(string num)
{
	if (num.size() < 3)return false;
	for (int l1 = 1; l1 < num.size(); ++l1) {
		string s1 = num.substr(0, l1);
		if (s1[0] == '0' && l1 > 1) break;
		long long num1 = stoll(s1);
		for (int l2 = l1 + 1; l2 < num.size() - l1; ++l2) {
			string s2 = num.substr(l1, l2);
			if (s2[0] == '0' && l2 > 1) break;
			long long num2 = stoll(s2);
			if (additiveNumBacktrack(num1, num2, num, l1 + l2)) return true;
		}
	}
	return false;
}
bool solution::additiveNumBacktrack(long long num1, long long num2, string s, int start)
{
	if (start == s.size()) return true;
	for (int l3 = 1; l3 <= s.size() - start; ++l3) {
		string s3 = s.substr(start, l3);
		if (s3[0] == '0' && l3 > 1) break;
		long long num3 = stoll(s3);
		if (num1 + num2 == num3 && additiveNumBacktrack(num2, num3, s, start + l3)) return true;
	}
	return false;
}

//No 313 Super Ugly Number
int solution::nthSuperUglyNumber(int n, vector<int> & primes)
{
	if (n <= 1) return 1;
	sort(primes.begin(), primes.end());
	vector<int>count(primes.size(), 0);
	vector<int> dp(n);
	dp[0] = 1;
	for (int i = 1; i < n; ++i) {
		int tmp = dp[count[0]] * primes[0];
		int id = 0;
		for (int j = 1; j < primes.size(); ++j) {
			if (dp[count[j]] * primes[j] < tmp) {
				tmp = dp[count[j]] * primes[j];
				id = j;
			}
		}
		dp[i] = tmp;
		for (int j = 0; j < primes.size(); ++j) {
			if (dp[i] == dp[count[j]] * primes[j]) {
				count[j]++;
			}
		}
	}
	return dp[n - 1];
}

//No 326  Power of Three
bool solution::isPowerOfThree(int n)
{
	if (n <= 0) return false;
	while (n > 1) {
		if (n % 3 != 0) return false;
		else {
			n = n / 3;
		}
	}
	return true;
}

//No 318 Maximum Product of Word Lengths
int solution::maxProductWordLength(vector<string> & words)
{
	if (words.size() <= 1) return 0;
	vector<int>mask(words.size(), 0);
	long long res = 0;
	for (int i = 0; i < words.size(); ++i) {
		for (auto c : words[i]) {
			mask[i] |= (1 << (c - 'a'));
		}
		for (int j = 0; j < i; ++j) {
			if (!(mask[i] & mask[j])) {
				res = max(res, (long long)(words[i].size() * words[j].size()));
			}
		}
	}
	return res;
}

//No 328 Odd Even Linked List
ListNode* solution::oddEvenList(ListNode * head)
{
	if (head == nullptr || head->next == nullptr || head->next->next == nullptr) return head;
	ListNode * evenHelper = new ListNode(0);
	ListNode * oddP = head, *evenP = head->next, *helperP = evenHelper;
	while (evenP != nullptr && oddP != nullptr) {
		oddP->next = evenP->next;
		helperP->next = evenP;
		helperP = helperP->next;
		evenP->next = nullptr;
		if (oddP->next) oddP = oddP->next;
		else break;
		if (oddP->next)evenP = oddP->next;
		else break;
	}
	oddP->next = evenHelper->next;
	evenHelper->next = nullptr;
	return head;
}

//No 319 Bulb Switcher
int solution::bulbSwitch(int n)
{
	if (n <= 0) return 0;
	if (n == 1) return 1;
	int bulbsOnNum = 0;
	for (int i = 1; i * i <= n; ++i) {
		bulbsOnNum++;
	}
	return bulbsOnNum;
	//Time Limit Exceeded
	//int bulbsOnNum = 1;
	//for (int i = 2;i <= n;++i) {
	//  int count = 0;
	//  for (int j = 2;j <= i/2;++j) {
	//    if (i%j == 0) count++;
	//  }
	//  if (count % 2 == 1) bulbsOnNum++;
	//}
	//return bulbsOnNum;
	/**************************/

}

//No 337 House Robber III
int solution::rob3(TreeNode * root)
{
	vector<int> ans = rob3Dfs(root);
	return max(ans[0], ans[1]);
}
vector<int> solution::rob3Dfs(TreeNode * root)
{
	if (!root) return vector<int>(2, 0);
	vector<int>left = rob3Dfs(root->left);
	vector<int>right = rob3Dfs(root->right);
	vector<int>res(2, 0);
	res[0] = max(left[0], left[1]) + max(right[0], right[1]);
	res[1] = left[0] + right[0] + root->val;
	return res;
}


//No 322 Coin Change
int solution::coinChange(vector<int> & coins, int amount)
{
	vector<int> coinDp(amount + 1, amount + 1);
	coinDp[0] = 0;
	for (int i = 1; i <= amount; ++i) {
		for (int j = 0; j < coins.size(); ++j) {
			if (coins[j] <= i) {
				coinDp[i] = min(coinDp[i], coinDp[i - coins[j]] + 1);
			}
		}
	}
	return coinDp[amount] > amount ? -1 : coinDp[amount];
}

//No 342 Power of Four
bool solution::isPowerOfFour(int num)
{
	//if (num <= 0)return false;
	//while (num > 1) {
	//	if (num % 4 != 0) return false;
	//	else {
	//		num /= 4;
	//	}
	//}
	//return true;
	return num > 0 && num==pow(4, (int)(log(num) / log(4)));
}

//No 338 Counting Bits
vector<int> solution::countBits(int num)
{
	vector<int> ans(num + 1, 0);
	for (int i = 1; i <= num; ++i) {
		if (i % 2 == 0) ans[i] = ans[i >> 1];
		else ans[i] = ans[i >> 1] + 1;
	}
	return ans;
}

//No 309 Best Time to Buy and Sell Stock with Cooldown
int solution::maxProfit(vector<int> & prices)
{
	int sell = 0, sellPre = 0, buy = 999999999, sellOld = 0;
	for (int i = 0; i < prices.size(); ++i) {
		buy = min(buy, prices[i] - sellOld);
		sellOld = sell;
		sell = max(sellOld, prices[i] - buy);
	}
	return sell;

	//if (prices.empty()) return 0;
	//int f0 = -prices[0],f1=0,f2=0;
	//for (int i = 1; i < prices.size(); ++i) {
	//	int newf0 = max(f0, f2 - prices[i]);
	//	int newf1 = f0 + prices[i];
	//	int newf2 = max(f1, f2);
	//	f0 = newf0;
	//	f1 = newf1;
	//	f2 = newf2;
	//}
	//return max(f1, f2);
}

//No 327 Count of Range Sum
int solution::countRangeSum(vector<int> & nums, int lower, int upper)
{
	int ans = 0;
	long long sum = 0;
	multiset<long long>sums;
	sums.insert(0);
	for (int i = 0; i < nums.size(); ++i) {
		sum += nums[i];
		ans += distance(sums.lower_bound(sum - upper), sums.upper_bound(sum - lower));
		sums.insert(sum);
	}
	return ans;
}

//No 343 Integer Break
int solution::integerBreak(int n)
{
	if (n < 4) return n - 1;
	int ans = 1;
	while (n > 4) {
		n -= 3;
		ans *= 3;
	}
	ans *= n;
	return ans;
}

//No 324 Wiggle Sort II
void solution::wiggleSort(vector<int> & nums)
{
	vector<int>helper(nums.begin(), nums.end());
	sort(helper.begin(), helper.end());
	int i = 0, j = (helper.size() - 1) / 2, k = helper.size() - 1;
	while (i < nums.size()) {
		if (i % 2 == 0) {
			nums[i] = helper[j];
			j--;
		}
		else {
			nums[i] = helper[k];
			k--;
		}
		i++;
	}
}

//No 344 Reverse String
void solution::reverseString(vector<char> & s)
{
	int len = s.size();
	if (len <= 1) return;
	for (int i = 0; i < len / 2; ++i) {
		swap(s[i], s[len - 1 - i]);
	}
}

//No 345 Reverse Vowels of a String
string solution::reverseVowels(string s)
{
	int len = s.size();
	int i = 0, j = len - 1;
	while (i < j) {
		if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'A' || s[i] == 'E' || s[i] == 'I' || s[i] == 'O' || s[i] == 'U') {
			if (s[j] == 'a' || s[j] == 'e' || s[j] == 'i' || s[j] == 'o' || s[j] == 'u' || s[j] == 'A' || s[j] == 'E' || s[j] == 'I' || s[j] == 'O' || s[j] == 'U') {
				swap(s[i], s[j]);
				i++;
				j--;
			}
			else j--;
		}
		else i++;
	}
	return s;
}


//No 347 Top K Frequent Elements
vector<int> solution::topKFrequent(vector<int> & nums, int k)
{
	vector<int>ans;
	unordered_map<int, int>m;
	for (auto n : nums) ++m[n];
	priority_queue<pair<int, int>> heap;//by a heap
	for (auto i : m) heap.push({ i.second, i.first });//the frequency of appearance
	for (int i = 0; i < k; ++i) {
		ans.push_back(heap.top().second);
		heap.pop();
	}
	return ans;
}

//No 349 Intersection of Two Arrays
vector<int> solution::intersection(vector<int> & nums1, vector<int> & nums2)
{
	vector<int> ans;
	unordered_map<int, int>m;
	for (auto n1 : nums1) m[n1] = 1;
	for (auto n2 : nums2) {
		if (m.find(n2) != m.end()) {
			m.erase(n2);
			ans.push_back(n2);
		}
	}
	return ans;
}

//No 350 Intersection of Two Arrays II
vector<int> solution::intersect(vector<int> & nums1, vector<int> & nums2)
{
	vector<int> ans;
	unordered_map<int, int>m;
	for (auto n1 : nums1) m[n1]++;
	for (auto n2 : nums2) {
		if (m.find(n2) != m.end()) {
			if (m[n2] > 0) {
				ans.push_back(n2);
				m[n2]--;
			}
		}
	}
	return ans;
}

//No 334 Increasing Triplet Subsequence
bool solution::increasingTriplet(vector<int> & nums)
{
	int len = nums.size();
	if (len < 3) return false;
	int m1 = INT_MAX, m2 = INT_MAX;
	for (int i = 0; i < len; ++i) {
		if (m1 >= nums[i]) m1 = nums[i];
		else if (m2 >= nums[i]) m2 = nums[i];
		else return true;
	}
	return false;
}

//No 354 Russian Doll Envelopes
//brute force DP
//int solution::maxEnvelopes(vector<pair<int, int>>& envelopes)
//{
//  int n = envelopes.size();
//  if (n <= 1) return n;
//  sort(envelopes.begin(), envelopes.end(), comparePair);
//  vector<int> dp(n, 1);
//  int count = 1;
//  int w = envelopes[0].first, h = envelopes[0].second;
//  for (int i = 1;i < n;++i) {
//    for (int j = 0;j < i;++j) {
//      if (envelopes[i].first > envelopes[j].first && envelopes[i].second > envelopes[j].second) {
//        dp[i] = max(dp[i], dp[j] + 1);
//      }
//    }
//    count = max(count, dp[i]);
//  }
//  return count;
//}
//bool solution::comparePair(pair<int, int>& p1, pair<int, int>& p2)
//{
//  if (p1.first != p2.first) return p1.first < p2.first;
//  else return p1.second < p2.second;
//}
/*****************************************************/
int solution::maxEnvelopes(vector<pair<int, int>> & envelopes) {
	int n = envelopes.size();
	if (n <= 1) return n;
	vector<int>dp;
	sort(envelopes.begin(), envelopes.end(), [](const pair<int, int> & p1, const pair<int, int> & p2) {
		return p1.first < p2.first || (p1.first == p2.first && p1.second > p2.second);
		});
	//for (auto p : envelopes) { cout << p.first << "," << p.second << endl; }
	for (auto p : envelopes) {
		auto it = lower_bound(dp.begin(), dp.end(), p.second);
		if (it == dp.end()) dp.push_back(p.second);
		else *it = p.second;
	}
	/*for (auto n : dp) { cout << n << " "; }
	cout << endl;*/
	return dp.size();
}


//No 357 Count Numbers with Unique Digits
int solution::countNumbersWithUniqueDigits(int n)
{
	if (n == 0) return 1;
	int ans = 10, cnt = 9;
	for (int i = 2; i <= n; ++i) {
		cnt *= (11 - i);
		ans += cnt;
	}
	return ans;
}

//No 363 Max Sum of Rectangle No Larger Than K
int solution::maxSumSubmatrix(vector<vector<int>> & matrix, int k)
{
	if (matrix.empty() || matrix[0].empty()) return 0;
	int row = matrix.size(), col = matrix[0].size(), ans = INT_MIN;
	for (int i = 0; i < col; ++i) {
		vector<int>sum(row, 0);
		for (int j = i; j < col; ++j) {//col start from i, end to j
			for (int r = 0; r < row; ++r) {
				sum[r] += matrix[r][j];
			}
			cout << j << ": sum";
			for (auto sr : sum) cout << " " << sr;
			cout << endl;
			int curSum = 0, curMax = INT_MIN;
			set<int>s{ 0 };
			for (auto sr : sum) {
				curSum += sr;
				auto it = s.lower_bound(curSum - k);//S[i-j]=S[i]-S[j]<=k
				if (it != s.end()) curMax = max(curMax, curSum - *it);//it==s.end()means curSum-k>0, this sum doesn't satisfy the requirment
				s.insert(curSum);
				cout << "curSum: " << curSum << endl;
			}
			ans = max(ans, curMax);
		}
	}
	return ans;
}

//No 365 Water and Jug Problem
bool solution::canMeasureWater(int x, int y, int z)
{
	//if such integer m and n exist to make z = m*x + n*y, we can get z litres water by those jugs
	//according to The Bézout's Identity, z=m*x+n*y has a solution if and only if z is a mutiple of the gcd(x,y)
	if (z == 0)return true;
	if (z<0 || z>x + y) return false;
	if ((x == 0 && y != z) || (y == 0 && x != z)) return false;
	//get gcd(x,y)
	int l = max(x, y), s = min(x, y);
	while (s > 0) {
		int t = s;
		s = l % s;
		l = t;
	}
	if (z % l == 0) return true;
	else return false;
}

//No 367 Valid Perfect Square
bool solution::isPerfectSquare(int num)
{
	num = (long long)num;
	if (num <= 0) return false;
	long long i = 0;
	while (i * i < num)++i;
	return i * i == num;
}

//No 368 Largest Divisible Subset
vector<int> solution::largestDivisibleSubset(vector<int> & nums)
{
	if (nums.empty()) return vector<int>();
	sort(nums.begin(), nums.end());
	vector<int>dp(nums.size(), 0), parent(nums.size(), 0), ans;
	int mx = 0, mxIdx = 0;
	for (int i = nums.size() - 1; i > -1; --i) {
		for (int j = i; j < nums.size(); ++j) {
			if (nums[j] % nums[i] == 0 && dp[i] < dp[j] + 1) {
				dp[i] = dp[j] + 1;
				parent[i] = j;
				if (dp[i] > mx) {
					mx = dp[i];
					mxIdx = i;
				}
			}
		}
	}
	for (int i = 0; i < mx; ++i) {
		ans.push_back(nums[mxIdx]);
		mxIdx = parent[mxIdx];
	}
	return ans;
}

//No 371 Sum of Two Integers
int solution::getSum(int a, int b)
{
	//if (b == 0) return a;
	//int sum = a^b;
	//int carry = (a & b & 0x7FFFFFFF) << 1;
	//return getSum(sum, carry);
	int sum = a ^ b;
	int carry = (a & b & 0x7FFFFFFF) << 1;
	while (carry) {
		int t = sum;
		sum = sum ^ carry;
		carry = (t & carry & 0x7FFFFFFF) << 1;//overflow
	}
	return sum;
}

//No 372 Super Pow
int solution::superPow(int a, vector<int> & b)
{
	int base = 1337;
	if (a == 0) return 0;
	if (b.empty()) return 1;
	long long ans = 1;
	for (auto n : b) {
		ans = powMod(ans, 10, base) * powMod(a, n, base);
	}
	return ans % base;
}

int solution::powMod(int a, int k, int base)
{
	a %= base;
	int result = 1;
	for (int i = 0; i < k; ++i) {
		result = (result * a) % base;
	}
	return result;
}


//No 373 Find K Pairs with Smallest Sums
vector<pair<int, int>> solution::kSmallestPairs(vector<int> & nums1, vector<int> & nums2, int k)
{
	//brute force
	if (nums1.empty() || nums2.empty()) return vector<pair<int, int>>();
	multimap<int, pair<int, int>> mm;
	vector<pair<int, int>>ans;
	for (int i = 0; i < min((int)nums1.size(), k); ++i) {
		for (int j = 0; j < min((int)nums2.size(), k); ++j) {
			mm.insert({ nums1[i] + nums2[j],{nums1[i],nums2[j]} });
		}
	}
	auto itm = mm.begin();
	while (k > 0 && itm != mm.end()) {
		ans.push_back(itm->second);
		--k;
		++itm;
	}
	return ans;
}

//No 375 Guess Number Higher or Lower II
int solution::getMoneyAmount(int n)
{
	vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
	return moneyAcountHelper(1, n, dp);
}
int solution::moneyAcountHelper(int start, int end, vector<vector<int>> & dp)
{
	if (start >= end) return 0;
	if (dp[start][end] > 0) return dp[start][end];
	dp[start][end] = INT_MAX;
	for (int i = start; i <= end; ++i) {
		int t = i + max(moneyAcountHelper(start, i - 1, dp), moneyAcountHelper(i + 1, end, dp));
		dp[start][end] = min(dp[start][end], t);
	}
	return dp[start][end];
}


//No 376 Wiggle Subsequence
int solution::wiggleMaxLength(vector<int> & nums)
{
	if (nums.size() <= 1) return nums.size();
	int diff = nums[1] - nums[0];
	int ans = 1;
	if (diff != 0) ++ans;
	for (int i = 2; i < nums.size(); ++i) {
		if (nums[i] != nums[i - 1]) {
			if (diff * (nums[i] - nums[i - 1]) < 0) ++ans;
			diff = nums[i] - nums[i - 1];
		}
	}
	return ans;
}


//No 377 Combination Sum IV
int solution::combinationSum4(vector<int> & nums, int target)
{
	vector<int>dp(target + 1, 0);
	dp[0] = 1;
	for (int i = 1; i <= target; ++i) {
		for (int j = 0; j < nums.size(); ++j) {
			if (nums[j] <= i) dp[i] += dp[i - nums[j]];
		}
	}
	return dp[target];
}

//No 378 Kth Smallest Element in a Sorted Matrix
int solution::kthSmallest(vector<vector<int>> & matrix, int k)
{
	//brute force by a heap
	//priority_queue<int> heap;
	//for (auto r : matrix) {
	//  for (auto e : r) {
	//    heap.emplace(e);
	//    if (heap.size() > k) heap.pop();
	//  }
	//}
	//return heap.top();

	//binary search O(n*lgX) where X = max - min in the matrix
	int low = matrix[0][0], high = matrix.back().back(), mid;
	while (low < high) {
		int t = 0;//mid is tth smallest element in the matrix 
		int r = 0, c = matrix[0].size() - 1;//right-top
		mid = low + (high - low) / 2;
		while (r < matrix.size() && c >= 0) {
			if (mid >= matrix[r][c]) {//every element in row r is smaller than mid
				t += c + 1;
				++r;
			}
			else --c;//every element in column c is larger than mid
		}
		//cout << mid << " is No. " << t << endl;
		if (t < k)low = mid + 1;//kth is larger than mid
		else high = mid;//kth is smaller than mid
	}
	return low;
}

//No 383 Ransom Note
bool solution::canConstruct(string ransomNote, string magazine)
{
	vector<int> m(26, 0);
	for (auto c : magazine) {
		++m[c - 'a'];
	}
	for (auto c : ransomNote) {
		--m[c - 'a'];
	}
	for (auto i : m) {
		if (i < 0) return false;
	}
	return true;
}

//No 386 Lexicographical Numbers
vector<int> solution::lexicalOrder(int n)
{
	//vector<int> ans(n);
	//for (int i = 0;i < n;++i) {
	//  ans[i] = i + 1;
	//}
	//sort(ans.begin(), ans.end(), [](int a1, int a2) {return to_string(a1) < to_string(a2);});
	//return ans;
	vector<int>ans(n);
	int cur = 1;
	for (int i = 0; i < n; ++i) {
		ans[i] = cur;
		if (cur * 10 <= n) cur *= 10;
		else {
			if (cur >= n) cur /= 10;
			cur += 1;
			while (cur % 10 == 0) cur /= 10;
		}
	}
	return ans;
}

//No 387 First Unique Character in a String
int solution::firstUniqChar(string s)
{
	vector<int> m(26, 0);
	for (auto c : s) ++m[c - 'a'];
	for (int i = 0; i < s.size(); ++i) {
		if (m[s[i] - 'a'] == 1) return i;
	}
	return -1;
}

//No 389 Find the Difference
char solution::findTheDifference(string s, string t)
{
	char ans = 0;
	for (auto c : s) ans ^= c;
	for (auto c : t) ans ^= c;
	return ans;
}

//No 871 Minimum Number of Refueling Stops
int solution::minRefuelStops(int target, int startFuel, vector<vector<int>> & stations)
{
	//greedy
	if (startFuel >= target) return 0;
	else if (stations.empty()) return -1;
	int cur = startFuel;
	priority_queue<int> f;
	int next = 0;
	int cnt = 0;
	while (next < stations.size() && cur >= stations[next][0] || !f.empty()) {
		//if the car can reach the next station, no need to refuel, save the fuel in a priority queue
		while ((next < stations.size() && cur >= stations[next][0])) f.push(stations[next++][1]);
		cur += f.top();
		f.pop();
		++cnt;
		if (cur >= target) return cnt;
	}
	return -1;
}

//No 390 Elimination Game
int solution::lastRemaining(int n)
{
	return helpLastRemaining(n, true);
}
int solution::helpLastRemaining(int n, bool l2r)
{
	if (n == 1) return 1;
	if (l2r) {
		return 2 * helpLastRemaining(n / 2, false);
	}
	else {
		return 2 * helpLastRemaining(n / 2, true) - 1 + n % 2;
	}
}



//No 392 Is Subsequence
bool solution::isSubsequence(string s, string t)
{
	int i = 0;
	for (int j = 0; j < t.size() && i < s.size(); ++j) {
		if (s[i] == t[j]) ++i;
	}
	return i == s.size();
}

//No 394 Decode String
string solution::decodeString(string s)
{
	string decodeRes = "";
	string val = "";
	//bool open = false;
	stack<string> decodeStack;
	for (int i = 0; i < s.size(); ++i) {
		if (isdigit(s[i])) { val += s[i]; }
		else if (s[i] == '[') {
			decodeStack.push(val);
			val = "";
			decodeStack.push(string(1, s[i]));
			//open = true;
		}
		else if (isalpha(s[i])) {
			decodeStack.push(string(1, s[i]));
		}
		else if (s[i] == ']') {
			string c = decodeStack.top();
			decodeStack.pop();
			string tmp = "";
			while (!decodeStack.empty() && c != "[") {
				tmp = c + tmp;
				c = decodeStack.top();
				decodeStack.pop();
			}
			c = decodeStack.top();
			decodeStack.pop();
			int multi = stoi(c);
			for (int j = 0; j < multi; ++j) {
				decodeRes += tmp;
			}
			decodeStack.push(decodeRes);
			decodeRes = "";
			//open = false;
		}
	}
	decodeRes = "";
	while (!decodeStack.empty()) {
		string tmp = decodeStack.top();
		decodeStack.pop();
		decodeRes = tmp + decodeRes;
	}
	return decodeRes;
}

//No 395 Longest Substring with At Least K Repeating Characters
int solution::longestSubstring(string s, int k)
{
	int ans = 0, i = 0, n = s.size();
	while (i + k <= n) {
		vector<int> m(26, 0);
		int mask = 0, maxIdx = 0;
		for (int j = i; j < n; ++j) {
			int t = s[j] - 'a';
			++m[t];
			if (m[t] < k) mask |= (1 << t);
			else mask &= (~(1 << t));
			if (mask == 0) {
				ans = max(ans, j - i + 1);
				maxIdx = j;
			}
		}
		i = maxIdx + 1;
	}
	return ans;
}

//No 393 UTF-8 Validation
bool solution::validUtf8(vector<int> & data)
{
	int cnt = 0;
	for (auto d : data) {
		if (cnt == 0) {
			if ((d >> 5) == 0b110) cnt = 1;
			else if ((d >> 4) == 0b1110) cnt = 2;
			else if ((d >> 3) == 0b11110) cnt = 3;
			else if (d >> 7) return false;
		}
		else {
			if ((d >> 6) != 0b10) return false;
			--cnt;
		}
	}
	return cnt == 0;
}

//No 396 Rotate Function
int solution::maxRotateFunction(vector<int> & A)
{
	long long F = 0, sum = 0, n = A.size();
	for (int i = 0; i < n; ++i) {
		F += i * A[i];
		sum += A[i];
	}
	//F(i) = F(i-1) + sum - n * A[n-i]
	long long maxSum = F;
	for (int i = 1; i < n; ++i) {
		F = F + sum - n * A[n - i];
		maxSum = max(maxSum, F);
	}
	return maxSum;
}

//No 397 Integer Replacement
int solution::integerReplacement(int n)
{
	if (n <= 3) return n - 1;
	int cnt = 2;
	while (n > 4) {
		if (n % 2 == 0) n /= 2;
		else if (n % 4 == 3) {
			n = n / 2 + 1;
			++cnt;
		}
		else n -= 1;
		++cnt;
	}
	return cnt;
}

//No 400 Nth Digit
int solution::findNthDigit(int n)
{
	long long digit = 1, start = 1;;
	long long num = 9;
	while (n > num * digit) {
		n -= num * digit;
		++digit;
		num *= 10;
		start *= 10;
	}
	//int k = (n-1) / digit;
	//int m = (n-1) % digit;

	start += (n - 1) / digit;
	cout << start << endl;
	return to_string(start)[(n - 1) % digit] - '0';
}

//No 399 Evaluate Division
vector<double> solution::calcEquation(vector<pair<string, string>> equations, vector<double> & values, vector<pair<string, string>> queries)
{
	unordered_map<string, unordered_map<string, double>> m;
	vector<double> ans;
	for (int i = 0; i < equations.size(); ++i) {
		m[equations[i].first][equations[i].second] = values[i];
		m[equations[i].second][equations[i].first] = 1.0 / values[i];
	}
	for (auto que : queries) {
		unordered_set<string> visited;
		double t = calEquhelper(que.first, que.second, visited, m);
		ans.push_back((t > 0.0) ? t : -1);
	}
	return ans;
}

double solution::calEquhelper(string up, string down, unordered_set<string> & visited, unordered_map<string, unordered_map<string, double>> & m)
{
	if (m[up].count(down)) return m[up][down];
	for (auto a : m[up]) {
		if (visited.count(a.first)) continue;
		visited.insert(a.first);
		double t = calEquhelper(a.first, down, visited, m);
		if (t > 0.0) return t * a.second;
	}
	return -1.0;
}

//No 404 Sum of Left Leaves
int solution::sumOfLeftLeaves(TreeNode * root)
{
	if (root == nullptr) return 0;
	int sum = 0;
	sumLeftLeavesCore(root, sum);
	return sum;
}

void solution::sumLeftLeavesCore(TreeNode * root, int& sum)
{
	if (root->left != nullptr) sumLeftLeavesCore(root->left, sum);
	if (root->left != nullptr && root->left->left == nullptr && root->left->right == nullptr) sum += root->left->val;
	if (root->right != nullptr) sumLeftLeavesCore(root->right, sum);
}


//No 25 Reverse Nodes in k-Group
ListNode * solution::reverseKGroup(ListNode * head, int k)
{
	ListNode* helper = new ListNode(-1), *curEnd = helper;
	ListNode * p = head, *kk = p;
	while (p != nullptr) {
		kk = p;
		int i = 1;
		while (kk != nullptr && i < k) {
			kk = kk->next;
			++i;
		}
		if (kk == nullptr) {
			curEnd->next = p;
			break;
		}
		ListNode* nextStep = kk->next;
		kk->next = nullptr;

		curEnd->next = reverseList(p, kk);
		curEnd = p;
		p = nextStep;
	}
	return helper->next;

}

ListNode* solution::reverseList(ListNode * start, ListNode * end)
{
	if (start == end) return end;
	ListNode * curHead = start, *nextHead = start->next;
	curHead->next = nullptr;
	while (nextHead != end) {
		ListNode* tmp = nextHead->next;
		nextHead->next = curHead;
		curHead = nextHead;
		nextHead = tmp;
	}
	end->next = curHead;
	return end;
}

//No 23 Merge k sorted Lists
ListNode* solution::mergeKLists(vector<ListNode*> & lists)
{
	ListNode* helper = nullptr;
	for (auto list : lists) {
		helper = mergeTwoList(helper, list);
	}
	return helper;
}
ListNode* solution::mergeTwoList(ListNode * h1, ListNode * h2)
{
	if (h1 == nullptr) return h2;
	if (h2 == nullptr) return h1;
	ListNode * newHead;
	if (h1->val < h2->val) {
		newHead = h1;
		newHead->next = mergeTwoList(h1->next, h2);
	}
	else {
		newHead = h2;
		newHead->next = mergeTwoList(h1, h2->next);
	}
}

//No 32 Longest Valid Parentheses
int solution::longestValidParentheses(string s)
{
	int len = s.size(), maxNum = 0;
	vector<int>dp(len, 0);
	for (int i = 1; i < len; ++i) {
		int j = i - 1 - dp[i - 1];
		if (s[i] == '(' || j < 0 || s[j] == ')') dp[i] = 0;
		else {
			dp[i] = dp[i - 1] + 2;
			if (j > 0) dp[i] += dp[j - 1];
			maxNum = max(maxNum, dp[i]);
		}
	}
	return maxNum;

	//int len = s.size(), maxNum = 0;
	//vector<int>dp(len + 1, 0);
	//for (int i = 1;i <= len;++i) {
	//  int j = i - 2 - dp[i - 1];
	//  if (s[i - 1] == '(' || j<0 || s[j] == ')') dp[i] = 0;
	//  else {
	//    dp[i] = dp[i - 1] + 2 + dp[j];
	//    maxNum = max(maxNum, dp[i]);
	//  }
	//}
	//return maxNum;
}
//No 406 Queue Reconstruction by Height
vector<vector<int>> solution::reconstructQueue(vector<vector<int>> & people)
{
	sort(people.begin(), people.end(), [](const vector<int> & v1, const vector<int> & v2) {return v1[0] > v2[0] || v1[0] == v2[0] && v1[1] < v2[1]; });
	vector<vector<int>>ans;
	for (int i = 0; i < people.size(); ++i) {
		if (people[i][1] == ans.size()) {
			ans.push_back(people[i]);
		}
		else {
			ans.insert(ans.begin() + people[i][1], people[i]);
		}
	}
	return ans;
}
//No 403
bool solution::canCross(vector<int> & stones)
{
	map<int, set<int>> dp;
	dp.clear();
	int u = 0, v = 0, w = 0;
	set<int> st;
	set<int>::iterator it;
	dp[0].insert(1);//从0号开始可以向前跳的步数
	for (int i = 0; i < stones.size(); ++i)st.insert(stones[i]);
	for (int i = 0; i < stones.size(); ++i) {
		u = stones[i];
		if (dp.find(u) == dp.end())continue;
		if (u == stones[stones.size() - 1])break;
		for (it = dp[u].begin(); it != dp[u].end(); ++it) {
			w = *it;//当前step
			v = u + w;//当前到达的石头的Unit
			if (st.find(v) != st.end() && v > u) {
				dp[v].insert(w - 1);
				dp[v].insert(w);
				dp[v].insert(w + 1);
			}
		}
	}
	return dp.find(u) != dp.end() && dp[u].size() > 0;
}
//No 409
int solution::longestPalindrome(string s)
{
	vector<int> charMap(256, 0);
	for (auto c : s) {
		charMap[c]++;
	}
	int count = 0;
	bool hasOdd = false;
	for (int i = 0; i < 256; ++i) {
		if (charMap[i] != 0) {
			if (charMap[i] % 2 == 0) count += charMap[i];
			else {
				hasOdd = true;
				count += charMap[i] - 1;
			}
		}
	}
	if (hasOdd) count += 1;
	return count;
}
//No 412
vector<string> solution::fizzBuzz(int n)
{
	vector<string> ans;
	string f = "Fizz", b = "Buzz";
	for (int i = 1; i <= n; ++i) {
		string tmp = "";
		if (i % 3 != 0 && i % 5 != 0) tmp = to_string(i);
		else {
			if (i % 3 == 0) tmp += f;
			if (i % 5 == 0) tmp += b;
		}
		ans.push_back(tmp);
	}
	return ans;
}



//No 410
bool solution::cansplit(vector<int> & nums, int value, int m)
{
	int len = nums.size();
	int i = 0;
	int sum = 0;
	int split_count = 0;
	for (i = 0; i < len; i++)
	{
		if (sum + nums[i] <= value)
		{
			sum += nums[i];
		}
		else
		{
			split_count++;
			sum = nums[i];
		}
		if (split_count == m)
		{
			return false;
		}
	}
	return true;
}

int solution::splitArray(vector<int> & nums, int m)
{
	int len = nums.size();
	int i = 0;
	int max = nums[0];
	long long sum = 0;
	for (i = 0; i < len; i++)
	{
		if (nums[i] > max)
		{
			max = nums[i];
		}
		sum += nums[i];
	}
	int left = max;
	int right = sum;
	int mid = 0;
	while (left < right)
	{
		mid = left + (right - left) / 2;

		if (cansplit(nums, mid, m))
		{
			right = mid;
		}
		else
		{
			left = mid + 1;
		}
	}
	return left;
}

//No 72 Edit Distance
int solution::minDistance(string word1, string word2)
{
	int len1 = word1.size(), len2 = word2.size();
	vector<vector<int>> distance(len1 + 1, vector<int>(len2 + 1, 0));
	for (int i = 1; i <= len1; ++i) distance[i][0] = i;
	for (int i = 1; i <= len2; ++i) distance[0][i] = i;
	for (int i = 1; i <= len1; ++i) {
		for (int j = 1; j <= len2; ++j) {
			int tmp = distance[i - 1][j - 1];
			if (word1[i - 1] != word2[j - 1]) tmp += 1;
			distance[i][j] = min(tmp, min(distance[i - 1][j], distance[i][j - 1]) + 1);
		}
	}
	return distance[len1][len2];
}

//No 239 Sliding Window Maximum
vector<int> solution::maxSlidingWindow(vector<int> & nums, int k)
{
	vector<int> ans;
	deque<int> maxIndexq;
	for (int i = 0; i < nums.size(); ++i) {
		if (!maxIndexq.empty() && maxIndexq.front() == i - k)maxIndexq.pop_front();
		while (!maxIndexq.empty() && nums[i] > nums[maxIndexq.back()]) {
			maxIndexq.pop_back();
		}
		maxIndexq.push_back(i);
		if (i >= k - 1)ans.push_back(nums[maxIndexq.front()]);
	}
	return ans;
}


//No 297 Serialize and Deserialize Binary Tree
string solution::serialize(TreeNode * root)
{
	ostringstream oss;
	queue<TreeNode*>q;
	if (root != nullptr) q.push(root);
	while (!q.empty()) {
		TreeNode* cur = q.front();
		q.pop();
		if (cur != nullptr) {
			oss << to_string(cur->val) << ' ';
			q.push(cur->left);
			q.push(cur->right);
		}
		else oss << '#' << ' ';
	}
	return oss.str();
}

TreeNode* solution::deserialize(string data)
{
	if (data.empty()) return nullptr;
	istringstream iss(data);
	queue<TreeNode*>q;
	string val;
	iss >> val;
	TreeNode* root = new TreeNode(stoi(val));
	TreeNode* cur = root;
	q.push(cur);
	while (!q.empty()) {
		TreeNode* tmp = q.front();
		q.pop();
		if (!(iss >> val)) break;
		if (val != "#") {
			cur = new TreeNode(stoi(val));
			q.push(cur);
			tmp->left = cur;
		}
		if (!(iss >> val)) break;
		if (val != "#") {
			cur = new TreeNode(stoi(val));
			q.push(cur);
			tmp->right = cur;
		}
	}
	return root;
}

//No 57 Insert Interval
vector<vector<int>> solution::insertIntervals(vector<vector<int>> & intervals, vector<int> & newInterval)
{
	vector<vector<int>> ans;
	int len = intervals.size(), cur = 0;
	for (int i = 0; i < len; ++i) {
		if (intervals[i][1] < newInterval[0]) {
			ans.push_back(intervals[i]);
			++cur;
		}
		else if (intervals[i][0] > newInterval[1]) {
			ans.push_back(intervals[i]);
		}
		else {
			newInterval[0] = min(newInterval[0], intervals[i][0]);
			newInterval[1] = max(newInterval[1], intervals[i][1]);
		}
	}
	ans.insert(ans.begin() + cur, newInterval);
	return ans;
}

//No 330 Patching Array
int solution::minPatches(vector<int> & nums, int n)
{
	long miss = 1, res = 0, id = 0;
	while (miss <= n) {
		if (id < nums.size() && nums[id] <= miss) {
			miss += nums[id++];
		}
		else {
			miss += miss;
			++res;
		}
	}
	return res;
}

//No 41 First Missing Positive
int solution::firstMissingPositive(vector<int> & nums)
{
	int len = nums.size();
	for (int i = 0; i < len; ++i) {
		while (nums[i] > 0 && nums[i] <= len && nums[nums[i] - 1] != nums[i]) {
			swap(nums[i], nums[nums[i] - 1]);
		}
	}
	for (int i = 0; i < len; ++i) {
		if (nums[i] != i + 1) return i + 1;
	}
	return len + 1;
}

//No 42 Trapping Rain Water
int solution::trapRainWater(vector<int> & height)
{
	int sumWater = 0, left = 0, right = height.size() - 1, cur = 0;
	while (left < right) {
		if (height[left] < height[right]) {
			cur = left;
			while (height[left] >= height[cur] && cur < right) {
				sumWater += height[left] - height[cur];
				++cur;
			}
			left = cur;
		}
		else {
			cur = right;
			while (height[right] >= height[cur] && cur > left) {
				sumWater += height[right] - height[cur];
				--cur;
			}
			right = cur;
		}
	}
	return sumWater;
}

//No 233 Number of Digit One
int solution::countDigitOne(int n)
{
	int res = 0;
	for (long m = 1; m <= n; m *= 10) {
		long a = n / m, b = n % m;
		res += (a + 8) / 10 * m;
		if (a % 10 == 1) res += b + 1;
	}
	return res;
}

//No 438 Find All Anagrams in a String
vector<int> solution::findAnagrams(string s, string p)
{
	vector<int> ans;
	vector<int>ms(256, 0), mp(256, 0);
	for (int i = 0; i < p.size(); ++i) {
		++ms[s[i]];
		++mp[p[i]];
	}
	if (ms == mp) ans.push_back(0);
	for (int i = p.size(); i < s.size(); ++i) {
		++ms[s[i]];
		--ms[s[i - p.size()]];
		if (ms == mp) ans.push_back(i - p.size() + 1);
	}
	return ans;
}

//No 494 Target Sum
int solution::findTargetSumWays(vector<int> & nums, int S)
{
	int len = nums.size();
	vector<unordered_map<int, int>>dp(len + 1);//dp[i][j]: nums[0] to nums[i-1] sum is j
	dp[0][0] = 1;
	for (int i = 0; i < len; ++i) {
		for (auto a : dp[i]) {
			int sum = a.first, cnt = a.second;
			dp[i + 1][sum + nums[i]] += cnt;
			dp[i + 1][sum - nums[i]] += cnt;
		}
	}
	return dp[len][S];
}

//No 739 Daily Temperatures
vector<int> solution::dailyTemperatures(vector<int> & T)
{
	vector<int> ans(T.size());
	stack<int>sId;
	int i = 0;
	while (i < T.size()) {
		while (!sId.empty() && T[i] > T[sId.top()]) {
			ans[sId.top()] = (i - sId.top());
			sId.pop();
		}
		sId.push(i);
		++i;
	}
	return ans;
}

//No 76 Minimum Window Substring
string solution::minWindow(string s, string t)
{
	vector<int> need(128, 0);
	for (auto c : t) need[c]++;
	int count = t.size();
	int begin = 0, end = 0;
	int head = 0;//start pos
	int len = INT_MAX;
	while (end < s.size()) {
		if (need[s[end++]]-- > 0) --count;
		while (count == 0) {
			if (end - begin < len) {
				len = end - begin;
				head = begin;//record currrent position and length
			}
			if (need[s[begin++]]++ == 0)++count;
		}
	}
	return len == INT_MAX ? "" : s.substr(head, len);
}

//No 140 Word Break II
vector<string> solution::wordBreak2(string s, vector<string> & wordDict)
{
	unordered_map<string, vector<string>> wordMap;
	return wordBreak2Rec(s, wordDict, wordMap);
}

vector<string> solution::wordBreak2Rec(string s, vector<string> & wordDict, unordered_map<string, vector<string>> & wordMap)
{
	if (wordMap.count(s)) return wordMap[s];
	if (s.empty()) return{ "" };
	vector<string> ans;
	for (auto w : wordDict) {
		if (s.substr(0, w.size()) != w) continue;
		vector<string> tmp = wordBreak2Rec(s.substr(w.size()), wordDict, wordMap);
		for (auto st : tmp) {
			ans.push_back(w + (st.empty() ? "" : " ") + st);
		}
	}
	wordMap[s] = ans;
	return wordMap[s];
}


//No 124 Binary Tree Maximum Path Sum
int solution::maxPathSum(TreeNode * root)
{
	int res = INT_MIN;
	maxPathSumRec(root, res);
	return res;
}
int solution::maxPathSumRec(TreeNode * node, int& res)
{
	if (!node) return 0;
	int left = max(maxPathSumRec(node->left, res), 0);
	int right = max(maxPathSumRec(node->right, res), 0);
	res = max(res, left + right + node->val);
	return max(left, right) + node->val;
}



//No 617 Merge Two Binary Trees
TreeNode * solution::mergeTrees(TreeNode * t1, TreeNode * t2)
{
	if (!t1) return t2;
	if (!t2) return t1;
	TreeNode* t = new TreeNode(t1->val + t2->val);
	t->left = mergeTrees(t1->left, t2->left);
	t->right = mergeTrees(t1->right, t2->right);
	return t;
}

//No 128 Longest Consecutive Sequence
int solution::longestConsecutive(vector<int> & nums)
{
	int ans = 0;
	unordered_set<int> hash(nums.begin(), nums.end());
	for (auto n : nums) {
		if (hash.find(n) != hash.end()) {
			int pre = n - 1, next = n + 1, tmp = 1;
			hash.erase(n);
			while (hash.find(pre) != hash.end()) {
				++tmp;
				hash.erase(pre);
				--pre;
			}
			while (hash.find(next) != hash.end()) {
				++tmp;
				hash.erase(next);
				++next;
			}
			ans = max(ans, next - pre - 1);
		}
	}
	return ans;
}

//No 437 Path Sum III
int solution::pathSum3(TreeNode * root, int sum)
{
	int nums = 0;
	vector<TreeNode*> out;
	pathSum3Dfs(root, 0, sum, nums, out);
	return nums;
}

void solution::pathSum3Dfs(TreeNode * node, int curSum, int target, int& nums, vector<TreeNode*> & out)
{
	if (!node) return;
	curSum += node->val;
	if (curSum == target) ++nums;
	out.push_back(node);
	int t = curSum;
	for (int i = 0; i < out.size() - 1; ++i) {
		t -= out[i]->val;
		if (t == target) ++nums;
	}
	if (node->left) pathSum3Dfs(node->left, curSum, target, nums, out);
	if (node->right) pathSum3Dfs(node->right, curSum, target, nums, out);
	out.pop_back();
}



//No 581 Shortest Unsorted Continuous Subarray
int solution::findUnsortedSubarray(vector<int> & nums)
{
	int n = nums.size(), start = -1, end = -2;
	int mn = nums[n - 1], mx = nums[0];
	for (int i = 1; i < n; ++i) {
		mx = max(mx, nums[i]);
		mn = min(mn, nums[n - 1 - i]);
		if (mx > nums[i])end = i;
		if (mn < nums[n - 1 - i])start = n - 1 - i;
	}
	return end - start + 1;
}

//No 647 Palindromic Substrings
int solution::countSubstrings(string s)
{
	int n = s.size(), res = 0;
	vector<vector<bool>> dp(n, vector<bool>(n, false));
	for (int i = n - 1; i >= 0; --i) {
		for (int j = i; j < n; ++j) {
			dp[i][j] = (s[i] == s[j]) && (j - i <= 2 || dp[i + 1][j - 1]);
			if (dp[i][j])++res;
		}
	}
	return res;
}

//No 329 Longest Increasing Path in a Matrix
int solution::longestIncreasingPath(vector<vector<int>> & matrix)
{
	if (matrix.empty() || matrix[0].empty()) return 0;
	int rows = matrix.size(), cols = matrix[0].size();
	vector<vector<int>> dp(rows, vector<int>(cols, 0));
	vector<vector<int>> dirs{ {1,0},{-1,0},{0,1},{0,-1} };
	int ans = 0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			ans = max(ans, longestIncPathDfs(matrix, i, j, dirs, dp));
		}
	}
	return ans;
}
int solution::longestIncPathDfs(vector<vector<int>> & matrix, int row, int col, vector<vector<int>> & dirs, vector<vector<int>> & dp)
{
	if (dp[row][col] > 0) return dp[row][col]; 
	else {
		dp[row][col] = 1;
		for (int i = 0; i < 4; ++i) {
			int dx = dirs[i][0], dy = dirs[i][1];
			if (row + dx >= 0 && row + dx < matrix.size() && col + dy >= 0 && col + dy<matrix[0].size() && matrix[row + dx][col + dy]>matrix[row][col]) {
				dp[row][col] = max(longestIncPathDfs(matrix, row + dx, col + dy, dirs, dp) + 1, dp[row][col]);
			}
		}
	}
	return dp[row][col];
}



//No 51 N-Queens
vector<vector<string>> solution::solveNQueens(int n)
{
	vector<vector<string>> ans;
	vector<int> colIds(n, 0);
	for (int j = 0; j < n; ++j) {
		colIds[0] = j;
		nQueensCore(n, 1, colIds, ans);
	}
	return ans;
}

bool solution::nQueensIsValid(int n, int k, vector<int> & colIds)
{
	for (int i = 0; i < k; ++i) {
		if (colIds[i] == colIds[k] || abs(colIds[k] - colIds[i]) == k - i) return false;
	}
	return true;
}

void solution::nQueensCore(int n, int k, vector<int> & colIds, vector<vector<string>> & ans)
{
	if (k >= n) {
		vector<string> oneSolu;
		for (int i = 0; i < n; ++i) {
			string tmp(n, '.');
			tmp[colIds[i]] = 'Q';
			oneSolu.push_back(tmp);
		}
		ans.push_back(oneSolu);
		return;
	}
	else {
		for (int j = 0; j < n; ++j) {
			colIds[k] = j;
			if (nQueensIsValid(n, k, colIds)) {
				nQueensCore(n, k + 1, colIds, ans);
			}
		}
	}
}

//No 543 Diameter of Binary Tree
int solution::diameterOfBinaryTree(TreeNode * root)
{
	int maxDia = 0;
	diameterOfBinaryTreeCore(root, maxDia);
	return max(maxDia - 1, 0);
}

int solution::diameterOfBinaryTreeCore(TreeNode * root, int& maxDia)
{
	if (!root) return 0;
	else {
		int left = diameterOfBinaryTreeCore(root->left, maxDia);
		int right = diameterOfBinaryTreeCore(root->right, maxDia);

		maxDia = max(maxDia, left + right + 1);
		return max(left, right) + 1;
	}
}



//No 621 Task Scheduler
int solution::leastInterval(vector<char> & tasks, int n)
{
	vector<int> numOfTasks(26, 0);
	for (auto c : tasks) {
		numOfTasks[c - 'A']++;
	}
	sort(numOfTasks.begin(), numOfTasks.end());
	int mx = numOfTasks[25], i = 25, len = tasks.size();
	while (i >= 0 && numOfTasks[i] == mx) --i;
	return max(len, (mx - 1) * (n + 1) + 25 - i);
}

//No 448 Find All Numbers Disappeared in an Array
vector<int> solution::findDisappearedNumbers(vector<int> & nums)
{
	vector<int> ans;
	for (int i = 0; i < nums.size(); ++i) {
		int id = abs(nums[i]) - 1;
		if (nums[id] > 0) nums[id] = -nums[id];
	}
	for (int i = 0; i < nums.size(); ++i) {
		if (nums[i] > 0) ans.push_back(i + 1);
	}
	return ans;
}

//No 301 Remove Invalid Parentheses
vector<string> solution::removeInvalidParentheses(string s)
{
	vector<string> ans;
	char pa[2] = { '(',')' };
	removeInvPaCore(s, 0, 0, pa, ans);
	return ans;
}

void solution::removeInvPaCore(string s, int lastI, int lastJ, char parentheses[], vector<string> & ans)
{
	int cnt = 0;
	for (int i = lastI; i < s.size(); ++i) {
		if (s[i] == parentheses[0]) ++cnt;
		if (s[i] == parentheses[1]) --cnt;
		if (cnt >= 0) continue;
		for (int j = lastJ; j <= i; ++j) {
			if (s[j] == parentheses[1] && (j == lastJ || s[j - 1] != parentheses[1])) {
				string t = s.substr(0, j) + s.substr(j + 1);
				removeInvPaCore(t, i, j, parentheses, ans);
			}
		}
		return;
	}
	string r;
	for (int i = s.size() - 1; i >= 0; --i) r.push_back(s[i]);
	if (parentheses[0] == '(') {
		char pa2[2] = { ')', '(' };
		removeInvPaCore(r, 0, 0, pa2, ans);
	}
	else ans.push_back(r);
}

//No 312 Burst Balloons
int solution::maxCoins(vector<int> & nums)
{
	int n = nums.size() + 2;
	vector<int> arr(n);
	arr[0] = arr[nums.size() + 1] = 1;
	for (int i = 1; i <= nums.size(); ++i) arr[i] = nums[i - 1];
	vector<vector<int>> dp(n, vector<int>(n));
	for (int k = 2; k < n; ++k) {
		for (int left = 0; left < n - k; ++left) {
			int right = left + k;
			for (int i = left + 1; i < right; ++i) {
				dp[left][right] = max(dp[left][right], arr[left] * arr[i] * arr[right] + dp[left][i] + dp[i][right]);
			}
		}
	}
	return dp[0][n - 1];
}

//No 429 N-ary Tree Level Order Traversal
vector<vector<int>> solution::levelOrder(MultiTreeNode* root)
{
	vector<vector<int>> ans;
	vector<int> tmp;
	queue<MultiTreeNode*> q;
	q.push(root);
	MultiTreeNode* last = root;
	MultiTreeNode* nextLast = root;
	while (!q.empty()) {
		MultiTreeNode* cur = q.front();
		q.pop();
		tmp.push_back(cur->val);
		if (!cur->children.empty()) {
			for (auto c : cur->children) q.push(c);
		}
		nextLast = q.back();
		if (cur == last) {
			ans.push_back(tmp);
			tmp.clear();
			last = nextLast;
		}
	}
	return ans;
}

//No 442. Find All Duplicates in an Array
vector<int> solution::findDuplicates(vector<int> & nums)
{
	set<int>ans;
	int len = nums.size();
	for (int i = 0; i < len; ++i) {
		while (nums[i] != i + 1) {
			if (nums[i] == nums[nums[i]-1]) {
				ans.insert(nums[i]);
				break;
			}
      int tmp = nums[i];
      nums[i] = nums[nums[i] - 1];
      nums[tmp - 1] = tmp;
		}   
	}
  return vector<int>(ans.begin(),ans.end());
}

//No 289 Game of Life
void solution::gameOfLife(vector<vector<int>>& board)
{
	// 0 -> 0 : 0
	// 0 -> 1 : 3
	// 1 -> 0 : 2
	// 1 -> 1 : 1
	//1 & 2 means this is a live cell in current state
	//0 & 3 means this is a dead cell in current state
	//0 & 2 means this will be dead in next state
	//1 & 3 means this will be live in next state
	int rows = board.size(), cols = board[0].size();
	vector<int> dr = { -1,-1,-1,0,0,1,1,1 }, dc = { -1,0,1,-1,1,-1,0,1 };
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int cnt = 0;
			for (int k = 0; k < 8; ++k) {
				if (i + dr[k] >= 0 && i + dr[k] < rows && j + dc[k] >= 0 && j + dc[k] < cols && (board[i + dr[k]][j + dc[k]] == 1 || board[i + dr[k]][j + dc[k]] == 2)) ++cnt;
			}
			if (board[i][j] == 1) {
				if (cnt < 2 || cnt>3) board[i][j] = 2;
			}
			else {
				if (cnt == 3) board[i][j] = 3;
			}
		}
	}
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			board[i][j] = board[i][j] % 2;
		}
	}
}

//No 30
vector<int> solution::findSubstring(string s, vector<string>& words)
{
	if(s.empty()||words.empty()) return vector<int>();
	int n = s.size(), m = words.size(), wl = words[0].size();
	unordered_map<string, int>mp;
	vector<int>ans;
	for (int i = 0; i < m; ++i) {
		mp[words[i]]++;
	}
	for (int i = 0; i < n - m * wl + 1; ++i) {
		unordered_map<string, int>seen;
		int j = 0;
		for (j = 0; j < m; ++j) {
			string cur = s.substr(i + j * wl, wl);
			if (mp.find(cur) == mp.end()) break;
			else {
				seen[cur]++;
				if (seen[cur] > mp[cur]) break;
			}
		}
		if (j == m)ans.push_back(i);
	}
	return ans;
}

//No 115 Distinct Subsequences
int solution::numDistinct(string s, string t)
{
	/*if (t.empty()) return 1;
	else if (s.empty()) return 0;
	int m = s.size(), n = t.size();
	vector<vector<uint64_t>>dp(m + 1, vector<uint64_t>(n + 1, 0));
	for (int i = 0; i <= m; ++i) dp[i][0] = 1;
	for (int j = 1; j <= n; ++j)dp[0][j] = 0;
	for (int i = 1; i <= m; ++i) {
		for (int j = 1; j <= n; ++j) {
			if (j > i)dp[i][j] = 0;
			else {
				if (s[i-1] == t[j-1])dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
				else dp[i][j] = dp[i - 1][j];
			}
		}
	}
	return dp[m][n];*/
	if (t.empty()) return 1;
	else if (s.empty()) return 0;
	int m = s.size(), n = t.size();
	uint64_t left = 0, cur = 0,last=0;
	vector<uint64_t>dp(m + 1, 1);
	for (int j = 1; j <= n; ++j) {
		for (int i = j; i <= m; ++i) {
			if (s[i - 1] == t[j - 1]) cur = dp[i - 1] + left;
			else cur = left;
			dp[i - 1] = left;
			left = cur;
		}
		dp[m] = left;
		left = 0;
	}
	return dp[m];
}

//No 149 Max Points on a Line
int solution::maxPoints(vector<vector<int>>& points)
{
	if (points.size() < 3) return points.size();
	int maxP = 0,num=points.size();
	for (int i = 0; i < num; ++i) {
		int same = 1;
		for (int j = i + 1; j < num; ++j) {
			int count = 0;
			if (points[i][0] == points[j][0] && points[i][1] == points[j][1]) ++same;
			else {
				++count;
				int64_t dx = points[i][0] - points[j][0];
				int64_t dy = points[i][1] - points[j][1];
				for (int k = j + 1; k < num; ++k) {
					if(dx*(points[i][1]-points[k][1]) == dy*(points[i][0]-points[k][0])) ++count;
				}
				maxP = max(maxP, same + count);
			}
		}
	}
	return maxP;
}

//No 135 Candy
int solution::candy(vector<int>& ratings)
{//greedy
	int sum = 0, len=ratings.size();
	vector<int> c(len, 0);
	//left to right, make sure child with higher rating have more candies than child on the left
	for (int i = 0; i < len-1; ++i) {
		if (ratings[i + 1] > ratings[i]) c[i+1] = c[i] + 1;
	}
	//right to left
	for (int i = len-1; i >0 ; --i) {
		if (ratings[i - 1] > ratings[i] && c[i-1]<(c[i]+1)) c[i - 1] = c[i] + 1;
	}
	//add all number of candies
	//the extra 1 -> every one has at least one candy
	for (auto n : c) sum += n + 1;
	return sum;
}

//No 417 Pacific Atlantic Water Flow
void solution::pacificAtlanticDFS(vector<vector<int>>& matrix, int x, int y, vector<vector<bool>>& visit, int pre) {
	int n = matrix.size(), m = matrix[0].size();
	if (x < 0 || x >= n || y < 0 || y >= m || matrix[x][y] < pre) return;
	visit[x][y] = true;
	pacificAtlanticDFS(matrix, x - 1, y, visit, matrix[x][y]);
	pacificAtlanticDFS(matrix, x + 1, y, visit, matrix[x][y]);
	pacificAtlanticDFS(matrix, x, y - 1, visit, matrix[x][y]);
	pacificAtlanticDFS(matrix, x, y + 1, visit, matrix[x][y]);
}
vector<vector<int>> solution::pacificAtlantic(vector<vector<int>>& matrix)
{
	vector<vector<int>> ans;
	if (matrix.empty() || matrix[0].empty()) return ans;
	int n = matrix.size(), m = matrix[0].size();
	vector<vector<bool>> pacific(n, vector<bool>(m, false));
	vector<vector<bool>> atlantic(n, vector<bool>(m, false));
	for (int i = 0; i < n; ++i) {
		pacificAtlanticDFS(matrix, i, 0, pacific, matrix[i][0]);
		pacificAtlanticDFS(matrix, i, m-1, atlantic, matrix[i][m-1]);
	}
	for (int i = 0; i < m; ++i) {
		pacificAtlanticDFS(matrix, 0, i, pacific, matrix[0][i]);
		pacificAtlanticDFS(matrix, n-1, i, atlantic, matrix[n-1][i]);
	}
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			if (pacific[i][j] && atlantic[i][j]) ans.push_back({ i,j });
		}
	}
	return ans;
}

//No 468 Validate IP Address
bool solution::ifIpv4(string ip) {
	vector<char>tmp;
	int i = 0, len = ip.size(), count = 0;
	for (int i = 0; i < len; ++i) {
		if (ip[i] == '.') {
			if (tmp.empty() || tmp.size() > 3) return false;
			int cur = 0;
			for (auto t : tmp) {
				cur = cur * 10 + t - '0';
			}
			if (cur >= 0 && cur < 256) {
				++count;
				tmp.clear();
			}
			else return false;
		}
		else if (ip[i] >= '0' && ip[i] <= '9') {
			tmp.push_back(ip[i]);
			if (tmp.size() > 1 && tmp[0] == '0') return false;
		}
		else return false;
	}
	int cur = 0;
	if (tmp.empty()) return false;
	for (auto t : tmp) {
		cur = cur * 10 + t - '0';
	}
	if (cur >= 0 && cur < 256) {
		++count;
	}
	else return false;
	return count == 4;
}
bool solution::ifIpv6(string ip) {
	vector<char>tmp;
	int i = 0, len = ip.size(), count = 0;
	for (int i = 0; i < len; ++i) {
		if (ip[i] == ':') {
			if (tmp.empty() || tmp.size() > 4) return false;
			else {
				++count;
				tmp.clear();
			}
		}
		else if ((ip[i] >= '0' && ip[i] <= '9') || (ip[i] >= 'a' && ip[i] <= 'f') || (ip[i] >= 'A' && ip[i] <= 'F')) tmp.push_back(ip[i]);
		else return false;
	}
	if (tmp.size() > 4) return false;
	else ++count;
	return count == 8;
}

string solution::validIPAddress(string IP) {
	int s = -1;
	for (auto c : IP) {
		if (c == '.') {
			s = 4;
			break;
		}
		else if (c == ':') {
			s = 6;
			break;
		}
	}
	if (s == 4 && ifIpv4(IP)) return "IPv4";
	else if (s == 6 && ifIpv6(IP)) return "IPv6";
	else return "Neither";
}

//No 275 H-Index II
int solution::hIndex2(vector<int>& citations)
{
	if(citations.empty()) return 0;
	int n = citations.size();
	int left = 0, right = n - 1;
	while (left <= right) {
		int mid = (right - left) / 2 + left;
		if (citations[mid] == n - mid) return n - mid;
		else if (citations[mid] > n - mid) right = mid - 1;
		else left = mid + 1;
	}
	return n - left;
}

//No 125
bool solution::isPalindromeString(string s)
{
	if (s.empty()) return true;
	int len = s.size(), i = 0, j = len - 1;
	while (i < j) {
		if (!isalnum(s[i]))++i;
		else if (!isalnum(s[j]))--j;
		else {
			if (s[i] == s[j] || (isalpha(s[i]) && isalpha(s[j]) && (s[i] + 32 == s[j] || s[i] - 32 == s[j]))) {
				++i;
				--j;
			}
			else return false;
		}
	}
	return true;
}

//No 1044
string solution::longestDupSubstring(string S)
{
	return "";
}

//No 10. Regular Expression Matching
bool solution::isMatch(string s, string p)
{
	if (s.empty() && p.empty()) return true;
	vector<vector<bool>> dp(s.size() + 1, vector<bool>(p.size() + 1, false));
	dp[0][0] = true;
	for (int j = 1; j <= p.size(); ++j) {
		if (p[j - 1] == '*') dp[0][j] = dp[0][j - 2];
	}
	for (int i = 1; i <= s.size(); ++i) {
		for (int j = 1; j <= p.size(); ++j) {
			if (s[i - 1] == p[j - 1] || p[j - 1] == '.') dp[i][j] = dp[i - 1][j - 1];
			else if (p[j - 1] == '*') {
				dp[i][j] = dp[i][j - 2];
				if (p[j - 2] == '.' || p[j - 2] == s[i - 1])dp[i][j] = dp[i][j] || dp[i - 1][j];
			}
		}
	}
	return dp[s.size()][p.size()];
}
//No 16 three sum closest
int solution::threeSumClosest(vector<int>& nums, int target)
{
	if (nums.size() < 3)return 0;
	sort(nums.begin(), nums.end());
	int gap = INT_MAX, ans = 0;
	for (int i = 0; i < nums.size() - 2; ++i) {
		int j = i + 1, k = nums.size() - 1;
		while (j < k) {
			int tmp = nums[i] + nums[j] + nums[k];
			if (abs(tmp - target) < gap) {
				gap = abs(tmp - target);
				ans = tmp;
			}
			if (tmp == target)return tmp;
			else if (tmp < target) ++j;
			else --k;
		}
	}
	return ans;
}

//No 332 Reconstruct Itinerary
vector<string> solution::findItinerary(vector<vector<string>>& tickets)
{
	// iterative solution
	/*vector<string> ans;
	if (tickets.empty()) return ans;
	unordered_map<string, multiset<string>> table;
	for (auto t : tickets) {
		table[t[0]].insert(t[1]);
	}

	if (!table.count("JFK")) return ans;
	stack<string>st;
	st.push("JFK");
	while (!st.empty()) {
		string cur = st.top();
		if (table[cur].empty()) {
			ans.insert(ans.begin(), cur);
			st.pop();
		}
		else {
			st.push(*(table[cur].begin()));
			table[cur].erase(table[cur].begin());
		}
	}
	return ans;*/

	// recursive solution
	unordered_map < string, priority_queue<string, vector<string>, greater<string>>> table;
	vector<string> ans;
	for (auto t : tickets) {
		table[t[0]].push(t[1]);
	}
	findItineraryDFS("JFK", ans, table);
	reverse(ans.begin(), ans.end());
	return ans;
}

void solution::findItineraryDFS(string cur, vector<string>& ans, unordered_map < string, priority_queue<string, vector<string>, greater<string>>>& table) {
	while (!table[cur].empty()) {
		string tmp = table[cur].top();	
		table[cur].pop();
		findItineraryDFS(tmp, ans, table);
	}
	ans.push_back(cur);
}


//No 718 Maximum Length of Repeated Subarray
int solution::findLength(vector<int>& A, vector<int>& B)
{
	int lenA = A.size(), lenB = B.size();
	if (lenA == 0 || lenB == 0) return 0;
	vector<vector<int>> dp(lenA + 1, vector<int>(lenB + 1, 0));
	//dp[i][j] means the max length of sub array that A end with A[i-1] and B end with B[j-1]
	int maxL = 0;
	for (int i = 1; i <= lenA; ++i) {
		for (int j = 1; j <= lenB; ++j) {
			if (A[i - 1] == B[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
			maxL = max(maxL, dp[i][j]);
		}
	}
	return maxL;
}

//No 441 Arrange Coins
int solution::arrangeCoins(int n)
{
	if(n==0) return 0;
	if (n == 1) return 1;
	int64_t i = 2;
	int64_t nn = n;
	while (i * (i + 1) / 2 <= n) ++i;
	return i - 1;
}

//No 97 Interleaving String
bool solution::isInterleave(string s1, string s2, string s3)
{
	int l1 = s1.size(), l2 = s2.size(), l3 = s3.size();
	if ((l1 + l2) != l3) return false;
	vector<vector<bool>>dp(l1 + 1, vector<bool>(l2));
	//dp[i][j] means if s3(0 - i+j-1) is formed by the interleaving of s1(0,i-1) and s2(0,j-1)
	dp[0][0] = true;
	for (int i = 1; i <= l1; ++i) dp[i][0] = dp[i - 1][0] && (s1[i - 1] == s3[i - 1]);
	for (int j = 1; j <= l2; ++j) dp[0][j] = dp[0][j - 1] && (s2[j - 1] == s3[j - 1]);
	for (int i = 1; i <= l1; ++i) {
		for (int j = 1; j <= l2; ++j) {
			dp[i][j] = (dp[i - 1][j] && (s1[i - 1] == s3[i - 1 + j])) || (dp[i][j - 1] && (s2[j - 1] == s3[j - 1 + i]));
		}
	}
	return dp[l1][l2];
}

//No 107 Binary Tree Level Order Traversal II
vector<vector<int>> solution::levelOrderBottom(TreeNode* root)
{
	vector<vector<int>>ans;
	vector<int>tmp;
	if (!root) return ans;
	queue<TreeNode*>q;
	q.push(root);
	TreeNode* last = root, *nLast = root;
	while (!q.empty()) {
		TreeNode* cur = q.front();
		q.pop();
		tmp.push_back(cur->val);
		if (cur->left)q.push(cur->left);
		if (cur->right)q.push(cur->right);
		nLast = q.back();
		if (cur == last) {
			ans.push_back(tmp);
			tmp.clear();
			last = nLast;
		}
	}
	reverse(ans.begin(), ans.end());
	return ans;
}

//No 108 Convert Sorted Array to Binary Search Tree
TreeNode* solution::sortedArrayToBST(vector<int>& nums)
{
	return sortedArrayToBSTCore(nums, 0, nums.size() - 1);
}

TreeNode* solution::sortedArrayToBSTCore(vector<int>& nums, int l, int r)
{
	if (l > r) return nullptr;
	int mid = (r - l) / 2 + l;
	TreeNode* root = new TreeNode(nums[mid]);
	root->left = sortedArrayToBSTCore(nums, l, mid - 1);
	root->right = sortedArrayToBSTCore(nums, mid + 1, r);
	return root;
}

//No 957 Prison Cells After N Days
vector<int> solution::prisonAfterNDays(vector<int>& cells, int N)
{
	vector<int>ans;
	//period is 14
	N = N % 14;
	if (N == 0)N = 14;
	while (N) {
		ans = prisonAfterNDaysNext(cells);
		cells = ans;
		--N;
	}
	return ans;
}
vector<int> solution::prisonAfterNDaysNext(vector<int>& cells)
{
	int len = cells.size();
	vector<int>res(8, 0);
	for (int i = 1; i <= 6; ++i) {
		if (cells[i - 1] == cells[i + 1]) res[i] = 1;
	}
	return res;
}

//No 837 New 21 Game
//conditional probablity
double solution::new21Game(int N, int K, int W)
{
	if (K == 0 || N >= K + W) return 1.0;
	vector<double>sum(K + W);
	sum[0] = 1.0;
	for (int i = 1; i < K + W; ++i) {
		int t = min(i - 1, K - 1);
		if (i <= W) sum[i] = sum[i - 1] + sum[t] / W;
		else sum[i] = sum[i - 1] + (sum[t] - sum[i - W - 1]) / W;
	}
	return (sum[N] - sum[K - 1]) / (sum[K + W - 1] - sum[K - 1]);
}

//No 463
int solution::islandPerimeter(vector<vector<int>>& grid)
{
	int ans = 0;
	int m = grid.size(), n = grid[0].size();
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 1) {
				int tmp = 4;
				if (i > 0 && grid[i - 1][j] == 1)--tmp;
				if (i < m - 1 && grid[i + 1][j] == 1)--tmp;
				if (j > 0 && grid[i][j - 1] == 1)--tmp;
				if (j < n - 1 && grid[i][j + 1] == 1)--tmp;
				ans += tmp;
			}
		}
	}
	return ans;
}

//No 15 3Sum
vector<vector<int>> solution::threeSum(vector<int>& nums)
{
	vector<vector<int>> ans;
	if (nums.size() < 3) return ans;
	sort(nums.begin(), nums.end());
	int i = 0;
	while (i < nums.size() - 2) {
		int target = -nums[i];
		int j = i + 1, k = nums.size() - 1;
		while (j < k) {
			if (nums[j] + nums[k] == target) {
				ans.push_back({ nums[i],nums[j],nums[k] });
				++j;
				--k;
				//avoid duplicate
				while (j < nums.size() && nums[j] == nums[j - 1])++j;
				while (k >= 0 && nums[k] == nums[k + 1])--k;
			}
			else if (nums[j] + nums[k] < target)++j;
			else --k;
		}
		++i;
		//avoid duplicate
		while (i < nums.size() && nums[i] == nums[i - 1])++i;
	}
	return ans;
}

//No 662 Maximum Width of Binary Tree
int solution::widthOfBinaryTree(TreeNode* root)
{
	if (!root)return 0;
	uint64_t ans = 0;
	queue<pair<TreeNode*, uint64_t>> q;
	q.push({ root,0 });
	TreeNode* last = root, * nextL = root;
	uint64_t left = 0, right = 0;
	bool newline = true;
	while (!q.empty()) {
		TreeNode* cur = q.front().first;
		uint64_t curNum = q.front().second;
		if (newline) left = curNum;
		right = curNum;
		q.pop();
		if (cur->left) {
			q.push({ cur->left,curNum * 2 + 1 });
		}
		if (cur->right) {
			q.push({ cur->right,curNum * 2 + 2 });
		}
		nextL = q.back().first;
		if (cur == last) {
			last = nextL;
			newline = true;
		}
		else newline = false;
		ans = max(ans, right - left + 1);
	}
	return (int)ans;
}

//No 430 Flatten a Multilevel Doubly Linked List
Node* solution::flatten(Node* head)
{
	if (!head || (!head->child && !head->next)) return head;
	Node* c = nullptr, * n = nullptr;
	if (head->child) {
		c = flatten(head->child);
		head->child = nullptr;
	}
	if (head->next) {
		n = flatten(head->next);
		head->next = nullptr;
	}
	if (c) {
		head->next = c;
		c->prev = head;
	}
	if (n) {
		Node* p = head;
		while (p->next != nullptr)p = p->next;
		p->next = n;
		n->prev = p;
	}
	return head;
}

//No 785
bool solution::isBipartite(vector<vector<int>>& graph)
{
	int len = graph.size();
	vector<int>color(len, 0);
	for (int i = 0; i < len; ++i) {
		if (color[i] == 0) {
			if (!isBipartiteDfs(graph, color, i, 1)) return false;
		}
	}
	return true;
}

bool solution::isBipartiteDfs(vector<vector<int>>& graph, vector<int>& color, int v, int c)
{
	color[v] = c;
	for (int i = 0; i < graph[v].size(); ++i) {
		if (color[graph[v][i]] == c) return false;
		if (color[graph[v][i]] == 0 && !isBipartiteDfs(graph, color, graph[v][i], -c)) return false;
	}
	return true;
}


//No 99 Recover Binary Search Tree
void solution::recoverTree(TreeNode* root)
{
	recoverBSTinorder(root);
	swap(first->val, second->val);
}
void solution::recoverBSTinorder(TreeNode* root)
{
	if (!root) return;
	recoverBSTinorder(root->left);
	if (!pre) pre = root;
	else {
		if (pre->val > root->val) {
			if (!first) first = pre;
			second = root;
		}
		pre = root;
	}
	recoverBSTinorder(root->right);
}

//No 1025 Divisor Game
bool solution::divisorGame(int N)
{
	//O(1)solution
	//if N is an odd number, factors of N are all odd numbers, Alice choose any of them (i.e. k), N-k will be an even number
	//then Bob can pick 1, leave another odd number N-k-1 to Alice
	//So Alice always gets an odd number while Bob always gets an even number, finally Bob will get 2, and choose 1 to leave 1 to Alice
	//therefore, Alice will lose the game if N is an odd number. On the other hand, Alice always wins when N is an even number
	return N % 2 == 0;

	//O(N^2) dp solution
	//find all k that 0<k<N and N%k==0
	//if there is a number k that Alice will lose at (N-k), which means Alice can take k first and Bob will lose at N-k, so Alice will win at N
	//if Alice wins at all the (N-k)s, which means no matter what Alice takes first, Bob always wins. Alice loses at N.

	//vector<bool>dp(N + 1);
	//dp[1] = false;
	//dp[2] = true;
	//for (int i = 3; i <= N; ++i) {
	//	bool cur = false;
	//	for (int j = 1; j < i; ++j) {
	//		if (i % j == 0) {
	//			if (dp[i - j] == false) {
	//				cur = true;
	//				break;
	//			}
	//		}
	//	}
	//	dp[i] = cur;
	//}
	//return dp[N];
}
//No 797 All Paths From Source to Target
vector<vector<int>> solution::allPathsSourceTarget(vector<vector<int>>& graph)
{
	vector<vector<int>>ans;
	vector<int>cur;
	cur.push_back(0);
	allPathsSourceTargetDFS(graph, graph.size(), 0, cur, ans);
	return ans;
}
void solution::allPathsSourceTargetDFS(vector<vector<int>>& graph, int n, int k, vector<int>& cur, vector<vector<int>>& ans)
{
	if (k == n - 1) {
		ans.push_back(cur);
		return;
	}
	else {
		for (auto i : graph[k]) {
			cur.push_back(i);
			allPathsSourceTargetDFS(graph, n, i, cur, ans);
			cur.pop_back();
		}
	}
}


//No 1531 String Compression II
int solution::getLengthOfOptimalCompression(string s, int k)
{
	int num = s.size() - k;//pick num charactors
	vector<vector<int>> dp(s.size() + 1, vector<int>(num + 1, 1e9));
	dp[s.size()][num] = 0;
	for (int i = s.size() - 1; i >= 0; --i) {
		for (int cnt = 0; cnt <= num; ++cnt) {
			int same = 0;
			for (int j = i; j < s.size(); ++j) {
				same += (s[j] == s[i]);
				if (same + cnt > num) break;
				int cal = (same <= 1) ? same : (same <= 9) ? 2 : (same <= 99) ? 3 : 4;
				dp[i][cnt] = min(dp[i][cnt], cal + dp[j + 1][cnt + same]);
			}
			dp[i][cnt] = min(dp[i][cnt], dp[i + 1][cnt]);
		}
	}
	return dp[0][0];
}

//No 632 Smallest Range Covering Elements from K Lists
vector<int> solution::smallestRange(vector<vector<int>>& nums)
{
	int left = 0, right = INT_MAX,len = nums.size();
	vector<int>next(len);
	auto cmp = [&](const int& u, const int& v) {return nums[u][next[u]] > nums[v][next[v]]; };
	priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);
	int minV = 0, maxV = INT_MIN;
	for (int i = 0; i < len; ++i) {
		pq.emplace(i);
		maxV = max(maxV, nums[i][0]);
	}
	while (true) {
		int row = pq.top();
		pq.pop();
		minV = nums[row][next[row]];
		if (maxV - minV < right - left) {
			left = minV;
			right = maxV;
		}
		if (next[row] == nums[row].size() - 1) break;
		++next[row];
		maxV = max(maxV, nums[row][next[row]]);
		pq.emplace(row);
	}
	return { left,right };
}

//No 520 Detect Capital
bool solution::detectCapitalUse(string word)
{
	if (word[0] >= 'A' && word[0] <= 'Z') {
		if (word.size() == 1) return true;
		bool allCap = (word[1] >= 'A' && word[1] <= 'Z') ? true : false;
		for (int i = 2; i < word.size(); ++i) {
			if ((allCap && word[i] >= 'a' && word[i] <= 'z') || (!allCap && word[i] >= 'A' && word[i] <= 'Z')) return false;
		}
	}
	else {
		for (int i = 1; i < word.size(); ++i) {
			if (word[i] >= 'a' && word[i] <= 'z') continue;
			else return false;
		}
	}
	return true;
}

//No 415 Add Strings
string solution::addStrings(string num1, string num2)
{
	int i = num1.size() - 1, j = num2.size() - 1, c = 0;
	string ans;
	while (i >= 0 || j >= 0 || c > 0) {
		int tmp = c;
		if (i >= 0) {
			tmp += num1[i] - '0';
			--i;
		}
		if (j >= 0) {
			tmp += num2[j] - '0';
			--j;
		}
		c = tmp / 10;
		tmp = tmp % 10;
		ans.push_back('0' + tmp);
	}
	reverse(ans.begin(), ans.end());
	return ans;
}

//No 336 Palindrome Pairs
vector<vector<int>> solution::palindromePairs(vector<string>& words)
{
	int n = words.size();
	for (const string& word : words) {
		wordsRev.push_back(word);
		reverse(wordsRev.back().begin(), wordsRev.back().end());
	}
	for (int i = 0; i < n; ++i) {
		indices.emplace(wordsRev[i], i);
	}

	vector<vector<int>> ret;
	for (int i = 0; i < n; i++) {
		int m = words[i].size();
		if (!m) {
			continue;
		}
		string wordView(words[i]);
		for (int j = 0; j <= m; j++) {
			if (isPalindrome336(wordView, j, m - 1)) {
				int left_id = findWord336(wordView, 0, j - 1);
				if (left_id != -1 && left_id != i) {
					ret.push_back({ i, left_id });
				}
			}
			if (j && isPalindrome336(wordView, 0, j - 1)) {
				int right_id = findWord336(wordView, j, m - 1);
				if (right_id != -1 && right_id != i) {
					ret.push_back({ right_id, i });
				}
			}
		}
	}
	return ret;
}

int solution::findWord336(const string& s, int left, int right)
{
	auto iter = indices.find(s.substr(left, right - left + 1));
	return iter == indices.end() ? -1 : iter->second;
}
bool solution::isPalindrome336(const string& s, int left, int right)
{
	int len = right - left + 1;
	for (int i = 0; i < len / 2; i++) {
		if (s[left + i] != s[right - i]) {
			return false;
		}
	}
	return true;
}


//No 696
int solution::countBinarySubstrings(string s)
{
	int last = 0, cur = 1, res = 0;
	for (int i = 1; i < s.size(); ++i) {
		if (s[i] == s[i - 1]) ++cur;
		else {
			last = cur;
			cur = 1;
		}
		if (last >= cur) res++;
	}
	return res;
}

//No 994 Rotting Oranges
int solution::orangesRotting(vector<vector<int>>& grid)
{
	if (grid.empty()) return 0;
	int m = grid.size(), n = grid[0].size();
	int fresh = 0;
	queue<pair<int, int>>q;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (grid[i][j] == 1) ++fresh;
			else if (grid[i][j] == 2)q.push({ i,j });
		}
	}
	int time = 0;
	vector<pair<int, int>> dirs = { {0, 1}, {1, 0}, {-1, 0}, {0, -1} };
	while (fresh != 0 && !q.empty()) {
		int qsize = q.size();
		for (int i = 0; i < qsize; ++i) {
			int curX = q.front().first, curY = q.front().second;
			q.pop();
			for (auto d : dirs) {
				int newX = curX + d.first, newY = curY + d.second;
				if (newX >= 0 && newX < m && newY >= 0 && newY < n && grid[newX][newY] == 1) {
					grid[newX][newY] = 2;
					--fresh;
					q.push({ newX,newY });
				}
			}
		}
		++time;
	}
	return fresh == 0 ? time : -1;
}

//No 123 Best Time to Buy and Sell Stock III
int solution::maxProfit3(vector<int>& prices)
{
	//int len = prices.size();
	//vector<vector<int>>dp(len, vector<int>(4, 0));
	////0:第一次买
	////1:第一次卖
	////2:第二次买
	////3:第二次卖
	//dp[0][0] = -prices[0];
	//dp[0][2] = INT_MIN;
	//for (int i = 1; i < len; ++i) {
	//	dp[i][0] = max(dp[i - 1][0], -prices[i]);
	//	dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
	//	dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] - prices[i]);
	//	dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] + prices[i]);
	//}
	//return max(dp[len - 1][1], dp[len - 1][3]);
	int buy1 = INT_MAX, buy2 = INT_MAX,pro1=0,pro2=0;
	for (auto p : prices) {
		buy1 = min(buy1, p);
		pro1 = max(pro1, p - buy1);
		buy2 = min(buy2, p - pro1);
		pro2 = max(pro2, p - buy2);
	}
	return pro2;
}

//No 1103 Distribute Candies to People
vector<int> solution::distributeCandies(int candies, int num_people)
{
	vector<int>ans(num_people);
	int k = 0, total = 0, round = (num_people)*(num_people+1)/2;
	while (total <= candies) {
		total += (k * num_people* num_people) + round;
		++k;
	}
	k--;
	int n = (k - 1) * k / 2;
	total = 0;
	if (k > 0) {
		for (int i = 0; i < num_people; ++i) {
			ans[i] = n * num_people + k*(i + 1);
			total += ans[i];
			cout << ans[i] << " ";
		}
	}
	cout << endl;
	int last = candies - total;
	int i = 0;
	while (last) {
		int tmp = k * num_people + i + 1;
		if (last >= tmp) {
			last -= tmp;
			ans[i] += tmp;
			++i;
		}
		else {
			ans[i] += last;
			break;
		}
	}
	return ans;
}

//No 824 Goat Latin
string solution::toGoatLatin(string S)
{
	stringstream ss(S);
	vector<string>input;
	string tmp;
	while (ss >> tmp) input.push_back(tmp);
	unordered_set<char>v = { 'a','e','i','o','u','A','E','I','O','U' };
	int len = input.size();
	for (int i = 0; i < len; ++i) {
		if (!v.count(input[i][0])) {
			char f = input[i][0];
			input[i] = input[i].substr(1);
			input[i] += f;
		}
		input[i] += "ma";
		for (int j = 0; j < i + 1; ++j) input[i] += 'a';
	}
	string ans;
	for (auto a : input) {
		ans += a;
		ans += ' ';
	}
	ans.pop_back();
	return ans;
}

//No 529 Minesweeper
vector<vector<char>> solution::updateBoard(vector<vector<char>>& board, vector<int>& click)
{
	vector<int>dirX = { 0, 1, 0, -1, 1, 1, -1, -1 }, dirY = { 1, 0, -1, 0, 1, -1, 1, -1 };
	int x = click[0], y = click[1];
	if (board[x][y] == 'M') {
		board[x][y] = 'X';
	}
	else {
		updateBoardDfs(board, x, y, dirX, dirY);
	}
	return board;
}

void solution::updateBoardDfs(vector<vector<char>>& board, int x, int y, vector<int>& dirX, vector<int>& dirY)
{
	int cnt = 0;
	for (int i = 0; i < 8; ++i) {
		int nx = x + dirX[i];
		int ny = y + dirY[i];
		if (nx < 0 || nx >= board.size() || ny < 0 || ny >= board[0].size()) { continue; }
		cnt+=(board[nx][ny]=='M');
	}
	if (cnt > 0) {
		board[x][y] = cnt + '0';
	}
	else {
		board[x][y] = 'B';
		for (int i = 0; i < 8; ++i) {
			int nx = x + dirX[i];
			int ny = y + dirY[i];
			if (nx < 0 || nx >= board.size() || ny < 0 || ny >= board[0].size()||board[nx][ny]!='E') { continue; }
			updateBoardDfs(board, nx, ny, dirX, dirY);
		}
	}
}


//No 679
bool solution::judgePoint24(vector<int>& nums)
{
	vector<double>numsd;
	for (auto n : nums) numsd.push_back(static_cast<double>(n));
	return recJudgePoint24(numsd);
}

bool solution::recJudgePoint24(vector<double>& nums)
{
	if(nums.empty())return false;
	if (nums.size() == 1) {
		return abs(nums[0] - 24.0) < 1e-6;
	}
	int len = nums.size();
	for (int i = 0; i < len; ++i) {
		for (int j = 0; j < len; ++j) {
			if (i != j) {
				vector<double> nums2;
				for (int k = 0; k < len; ++k) {
					if (k != i && k != j) nums2.push_back(nums[k]);
				}
				for (int k = 0; k < 4; ++k) {
					//0:+
					//1:*
					//2:-
					//3:/
					if (k<2 && i>j) continue;//commutative laws of addition and multiplication
					if (k == 0) nums2.push_back(nums[i] + nums[j]);
					else if (k == 1) nums2.push_back(nums[i] * nums[j]);
					else if (k == 2) nums2.push_back(nums[i] - nums[j]);
					else {
						if(nums[j]>=1e-6) nums2.push_back(nums[i] / nums[j]);
					}
					if (recJudgePoint24(nums2)) return true;
					nums2.pop_back();
				}
			}
		}
	}
	return false;
}

//No 17
vector<string> solution::letterCombinations(string digits)
{
	unordered_map<char, string> m{
						{'2', "abc"},
						{'3', "def"},
						{'4', "ghi"},
						{'5', "jkl"},
						{'6', "mno"},
						{'7', "pqrs"},
						{'8', "tuv"},
						{'9', "wxyz"}
	};
	vector<string>ans;
	string cur = "";
	if (digits.empty()) return ans;
	backtraceLetterComb(ans, cur, digits, 0, m);
	return ans;
}

void solution::backtraceLetterComb(vector<string>& ans, string cur, string digits, int id, unordered_map<char, string>& m)
{
	if (id == digits.size()) {
		ans.push_back(cur);
		return;
	}
	else {
		char d = digits[id];
		string tmp = m[d];
		for (auto c : tmp) {
			cur.push_back(c);
			backtraceLetterComb(ans, cur, digits, id + 1, m);
			cur.pop_back();
		}
	}
}

//No 557
string solution::reverseWords(string s)
{
	int len = s.size();
	int begin = 0, end;
	for (int i = 0; i < len + 1; ++i) {
		if (i == len || s[i] == ' ') {
			end = i;
			reverse(s.begin() + begin, s.begin() + end);
			begin = i + 1;
		}
	}
	return s;
}

//No 841
bool solution::canVisitAllRooms(vector<vector<int>>& rooms)
{
	int num = rooms.size();
	vector<bool>visited(num, false);
	queue<int> q;
	q.push(0);
	int count = 0;
	while (!q.empty()) {
		int cur = q.front();
		q.pop();
		if (!visited[cur]) {
			visited[cur] = true;
			++count;
			for (auto t : rooms[cur]) {
				if (!visited[t]) q.push(t);
			}
		}
	}
	return count == num;
}

//No 486
bool solution::PredictTheWinner(vector<int>& nums)
{
	int len = nums.size();
	vector<vector<int>>dp(len, vector<int>(len));
	for (int i = 0; i < len; ++i)dp[i][i] = nums[i];
	for (int i = len - 2; i >= 0; --i) {
		for (int j = i + 1; j < len; ++j) {
			dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
		}
	}
	return dp[0][len-1]>=0; 
}

//No 290
bool solution::wordPattern(string pattern, string str)
{
	istringstream in(str);
	unordered_map<char, string>m;
	unordered_map<char, string>::iterator it;
	int i = 0;
	for (string word; in >> word; ++i) {
		if (m.find(pattern[i]) != m.end()) {
			if (m[pattern[i]] != word) return false;
		}
		else {
			for (it = m.begin(); it != m.end(); ++it) {
				if (it->second == word) return false;
			}
			m[pattern[i]] = word;
		}
	}
	return i == pattern.size();
}

//No 299　Bulls and Cows
string solution::getHint(string secret, string guess)
{
	vector<int>nums(10, 0);
	int bulls = 0, cows = 0;
	for (int i = 0; i < secret.size(); ++i) {
		if (secret[i] == guess[i])++bulls;
		else {
			if (nums[(int)(secret[i] - '0')]++ < 0) cows++;
			if (nums[(int)(guess[i] - '0')]-- > 0) cows++;
		}
	}
	return to_string(bulls) + 'A' + to_string(cows) + 'B';
}

//No 637
vector<double> solution::averageOfLevels(TreeNode* root)
{
	vector<double> ans;
	if (!root) return ans;
	queue<TreeNode*> q;
	q.push(root);
	while (!q.empty()) {
		double sum = 0.0;
		int count = 0;
		int size = q.size();
		for (int i = 0; i < size; ++i) {
			TreeNode* node = q.front();
			q.pop();
			sum += (double)node->val;
			++count;
			if (node->left) q.push(node->left);
			if (node->right) q.push(node->right);
		}
		ans.push_back(sum / (double)count);
	}
	return ans;
}
//No 501
vector<int> solution::findMode(TreeNode* root)
{
	vector<int>res;
	if (!root) return res;
	TreeNode* pre = nullptr;
	int curT = 1, maxT = 0;
	findModeinorder(root, pre, curT, maxT, res);
	return res;
}
void solution::findModeinorder(TreeNode* root, TreeNode*& pre, int& curT, int& maxT, vector<int>& res)
{
	if (!root) return;
	findModeinorder(root->left, pre, curT, maxT, res);
	if (pre) curT = (root->val == pre->val) ? curT + 1 : 1;
	if (curT == maxT) res.push_back(root->val);
	else if (curT > maxT) {
		res.clear();
		res.push_back(root->val);
		maxT = curT;
	}
	pre = root;
	findModeinorder(root->right, pre, curT, maxT, res);
}



string solution::largestNumber(vector<int>& nums)
{
	vector<string>nums2;
	for (auto n : nums) {
		string tmp = to_string(n);
		nums2.push_back(tmp);
	}
	sort(nums2.begin(), nums2.end(), [](string s1, string s2) {return s1 + s2 > s2 + s1; });
	string ans = "";
	for (auto s : nums2) {
		ans += s;
	}
	int len = ans.size(), i = 0;
	while (i < len - 1 && ans[i] == '0') ++i;
	return ans.substr(i);
}


//No 117
TreeLinkNode* solution::getNext(TreeLinkNode* root)
{
	if (!root) return nullptr;
	else {
		if (root->left) return root->left;
		else if (root->right) return root->right;
		else if (root->next) return getNext(root->next);
	}
	return nullptr;
}

TreeLinkNode* solution::connect117(TreeLinkNode* root)
{
	if (!root) return root;
	if (root->left) {
		if (root->right) root->left->next = root->right;
		else root->left->next = getNext(root->next);
	}
	if (root->right) root->right->next = getNext(root->next);
	connect(root->right);
	connect(root->left);
	return root;
}

TreeNode* solution::insertIntoBST(TreeNode* root, int val)
{
	if (!root) return new TreeNode(val);
	if (root->val < val) {
		root->right = insertIntoBST(root->right, val);
	}
	else root->left = insertIntoBST(root->left, val);
	return root;
}

//No 121
int solution::maxProfit1(vector<int>& prices)
{
	int buy = INT_MAX, profit = 0;
	for (auto p : prices) {
		profit = max(profit, p - buy);
		buy = min(buy, p);
	}
	return profit;
}
//No 122
int solution::maxProfit2(vector<int>& prices)
{
	if (prices.size() <= 1) return 0;
	int max_profit = 0;
	for (int i = 1; i < prices.size(); ++i) {
		if (prices[i] > prices[i - 1])
			max_profit += prices[i] - prices[i - 1];
	}
	return max_profit;
}
//No 188
int solution::maxProfit4(int k, vector<int>& prices)
{
	int len = prices.size();
	if (k < 1 || len < 2) return 0;
	if (k >= len / 2) {
		int max_profit = 0;
		for (int i = 1; i < prices.size(); ++i) {
			if (prices[i] > prices[i - 1])
				max_profit += prices[i] - prices[i - 1];
		}
		return max_profit;
	}
	else {
		vector<vector<int>>dp(k, vector<int>(2, 0));
		for (int i = 0; i < k; ++i) dp[i][0] = INT_MIN;
		for (auto p : prices) {
			dp[0][0] = max(dp[0][0], -p);
			dp[0][1] = max(dp[0][1], dp[0][0] + p);
			for (int i = 1; i < k; ++i) {
				dp[i][0] = max(dp[i][0], dp[i - 1][1] - p);
				dp[i][1] = max(dp[i][1], dp[i][0] + p);
			}
		}
		return dp[k - 1][1];
	}
}
//No 714
int solution::maxProfitFee(vector<int>& prices, int fee)
{
	int cash = 0, hold = INT_MIN;
	for (auto p : prices) {
		cash = max(cash, hold + p - fee);
		hold = max(hold, cash - p);
	}
	return cash;

}

vector<vector<int>> solution::fourSum(vector<int>& nums, int target)
{
	set<vector<int>> ans;
	sort(nums.begin(), nums.end());
	int len = nums.size();
	if (len < 4) return {};
	for (int i = 0; i < len - 3; ++i) {
		for (int j = i + 1; j < len - 2; ++j) {
			int tar = target - nums[i] - nums[j];
			int k = j + 1, l = len - 1;
			while (k < l) {
				if (nums[k] + nums[l] == tar) {
					ans.insert({ nums[i],nums[j],nums[k],nums[l] });
					++k;
					--l;
				}
				else if (nums[k] + nums[l] < tar)++k;
				else --l;
			}
		}
	}
	return vector<vector<int>>(ans.begin(), ans.end());
}

//No 24 
ListNode* solution::swapPairs(ListNode* head)
{
	ListNode* helper = new ListNode(0);
	helper->next = head;
	ListNode* p = head, * pre = helper;
	while (p && p->next) {
		ListNode* q = p->next;
		pre->next = q;
		p->next = q->next;
		q->next = p;
		pre = p;
		p = p->next;
	}
	return helper->next;
}

//No 19
ListNode* solution::removeNthFromEnd(ListNode* head, int n)
{
	ListNode* p = head, * pn = head;
	ListNode* helper = new ListNode(0), * pp = new ListNode(0);
	helper->next = head;
	pp = helper;
	for (int i = 0; i < n; ++i) pn = pn->next;
	while (pn) {
		p = p->next;
		pp = pp->next;
		pn = pn->next;
	}
	pp->next = p->next;
	return helper->next;
}
