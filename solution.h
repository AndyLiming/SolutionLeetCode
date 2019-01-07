#ifndef _SOLUTION_
#define _SOLUTION_
/*this class contains solutions of problems in leetcode*/
/*after each declaration, there is an explain include problem number, difficulty and short description*/
class solution {
  //define solution functions
public:
  vector<vector<string>> groupAnagrams(vector<string>& strs);//No 49, medium
  //49. Group Anagrams: Given an array of strings, group anagrams together.

  double myPow(double x, int n);//No 50, medium
  //50. Pow(x, n): Implement pow(x, n), which calculates x raised to the power n (xn).

  vector<int> spiralOrder(vector<vector<int>>& matrix);
  //54. Spiral Matrix: Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

  bool canJump(vector<int>& nums);
  //55. Jump Game: Given an array of non-negative integers, you are initially positioned at the first index of the array.Each element in the array represents your maximum jump length at that position.
  //Determine if you are able to reach the last index.

  vector<Interval> merge(vector<Interval>& intervals);
  //56. Merge Intervals: Given a collection of intervals, merge all overlapping intervals.

  vector<vector<int>> generateMatrix(int n);
  //59. Spiral Matrix II: Given a positive integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.

  string getPermutation(int n, int k);
  //60. Permutation Sequence: 1-n permytation find kth

  ListNode* rotateRight(ListNode* head, int k);
  //61. Rotate List: Given a linked list, rotate the list to the right by k places, where k is non-negative.

  int uniquePaths(int m, int n);
  //62. Unique Paths: A robot is located at the top-left corner of a m x n grid
  //The robot can only move either down or right at any point in time
  //How many possible unique paths are there?

  int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);
  //63. Unique Paths II

  int minPathSum(vector<vector<int>>& grid);
  //64. Minimum Path Sum: Given a m x n grid filled with non-negative numbers, 
  //find a path from top left to bottom right which minimizes the sum of all numbers along its path.

  string simplifyPath(string path);
  //71. Simplify Path: Given an absolute path for a file (Unix-style), simplify it. 

  void setZeroes(vector<vector<int>>& matrix);
  //73. Set Matrix Zeroes: Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

  bool searchMatrix(vector<vector<int>>& matrix, int target);
  //74. Search a 2D Matrix: Write an efficient algorithm that searches for a value in an m x n matrix.
  //Integers in each row are sorted from left to right; The first integer of each row is greater than the last integer of the previous row.

  void sortColors(vector<int>& nums);
  //75. Sort Colors: Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, 
  //with the colors in the order red, white and blue.

  vector<vector<int>> combine(int n, int k);
  //77. Combinations: Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.

  vector<vector<int>> subsets(vector<int>& nums);
  //78. Subsets: Given a set of distinct integers, nums, return all possible subsets (the power set).

  bool exist(vector<vector<char>>& board, string word);
  //79. Word Search: Given a 2D board and a word, find if the word exists in the grid.

  int removeDuplicates(vector<int>& nums);
  //80. Remove Duplicates from Sorted Array II: Given a sorted array nums, remove the duplicates in-place such that duplicates appeared at most twice and return the new length.

  bool search(vector<int>& nums, int target);
  //81. Search in Rotated Sorted Array II: Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
  //You are given a target value to search. If found in the array return true, otherwise return false.

  ListNode* deleteDuplicates(ListNode* head);
  //82. Remove Duplicates from Sorted List II: Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.

  int largestRectangleArea(vector<int>& heights);
  //84. Largest Rectangle in Histogram: Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, 
  //find the area of largest rectangle in the histogram.

  int maximalRectangle(vector<vector<char>>& matrix);
  //85. Maximal Rectangle: Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

  ListNode* partition(ListNode* head, int x);
  //86. Partition List: Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
  //You should preserve the original relative order of the nodes in each of the two partitions.

  vector<int> grayCode(int n);
  //89. Gray Code: The gray code is a binary numeral system where two successive values differ in only one bit.

  vector<vector<int>> subsetsWithDup(vector<int>& nums);
  //90. Subsets II: Given a collection of integers that might contain duplicates, nums, return all possible subsets (the power set).

  int numDecodings(string s);
  //91. Decode Ways: Given a non-empty string containing only digits, determine the total number of ways to decode it.

  ListNode* reverseBetween(ListNode* head, int m, int n);
  //92. Reverse Linked List II: Reverse a linked list from position m to n. Do it in one-pass.

  vector<string> restoreIpAddresses(string s);
  //93. Restore IP Addresses: Given a string containing only digits, restore it by returning all possible valid IP address combinations.

  vector<int> inorderTraversal(TreeNode* root);
  //94. Binary Tree Inorder Traversal: Given a binary tree, return the inorder traversal of its nodes' values.

  vector<TreeNode*> generateTrees(int n);
  //95. Unique Binary Search Trees II: Given an integer n, generate all structurally unique BST's (binary search trees) that store values 1 ... n.

  int numTrees(int n);
  //96. Unique Binary Search Trees: Given n, how many structurally unique BST's (binary search trees) that store values 1 ... n?

  bool isValidBST(TreeNode* root);
  //98. Validate Binary Search Tree: Given a binary tree, determine if it is a valid binary search tree (BST).

  vector<vector<int>> levelOrder(TreeNode* root);
  //102. Binary Tree Level Order Traversal: Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

  vector<vector<int>> zigzagLevelOrder(TreeNode* root);
  //103. Binary Tree Zigzag Level Order Traversal: Given a binary tree, return the zigzag level order traversal of its nodes' values.
  //(ie, from left to right, then right to left for the next level and alternate between).

  TreeNode* buildTree_preIn(vector<int>& preorder, vector<int>& inorder);
  //105. Construct Binary Tree from Preorder and Inorder Traversal: Given preorder and inorder traversal of a tree, construct the binary tree.

  TreeNode* buildTree_inPos(vector<int>& inorder, vector<int>& postorder);
  //106. Construct Binary Tree from Inorder and Postorder Traversal: Given inorder and postorder traversal of a tree, construct the binary tree.

  TreeNode* sortedListToBST(ListNode* head);
  //109. Convert Sorted List to Binary Search Tree: Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.

  vector<vector<int>> pathSum(TreeNode* root, int sum);
  //113. Path Sum II: Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

  void flatten(TreeNode* root);
  //114. Flatten Binary Tree to Linked List: Given a binary tree, flatten it to a linked list in-place.

  void connect(TreeLinkNode *root);
  //116. Populating Next Right Pointers in Each Node: Populate each next pointer to point to its next right node. 
  //If there is no next right node, the next pointer should be set to NULL.

  void connect_nonPerfect(TreeLinkNode *root);
  //117. Populating Next Right Pointers in Each Node II: Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

  int minimumTotal(vector<vector<int>>& triangle);
  //120. Triangle: Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

  int ladderLength(string beginWord, string endWord, vector<string>& wordList);
  //127. Word Ladder: Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord

  int sumNumbers(TreeNode* root);
  //129. Sum Root to Leaf Numbers: Find the total sum of all root-to-leaf numbers.
  
  void solveSurronded(vector<vector<char>>& board);
  //130. Surrounded Regions: Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

  vector<vector<string>> partition(string s);
  //131. Palindrome Partitioning: Given a string s, partition s such that every substring of the partition is a palindrome.
  //Return all possible palindrome partitioning of s.

  UndirectedGraphNode *cloneGraph(UndirectedGraphNode *node);
  //133. Clone Graph: Given the head of a graph, return a deep copy (clone) of the graph.

  int canCompleteCircuit(vector<int>& gas, vector<int>& cost);
  //134. Gas Station: There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
  //You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). 
  //You begin the journey with an empty tank at one of the gas stations.
  //Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.

  int singleNumber(vector<int>& nums);
  //137. Single Number II: Given a non-empty array of integers, every element appears three times except for one, which appears exactly once. Find that single one.

  RandomListNode *copyRandomList(RandomListNode *head);
  //138. Copy List with Random Pointer

  bool wordBreak(string s, vector<string>& wordDict);
  //139. Word Break: Given a non-empty string s and a dictionary wordDict containing a list of non-empty words,
  //determine if s can be segmented into a space-separated sequence of one or more dictionary words.

  bool hasCycle(ListNode *head);
  //141. Linked List Cycle: Given a linked list, determine if it has a cycle in it. 

  ListNode *detectCycle(ListNode *head);
  //142. Linked List Cycle II: Given a linked list, return the node where the cycle begins. If there is no cycle, return null. 

  void reorderList(ListNode* head);
  //143. Reorder List: Given a singly linked list L: L0¡úL1¡ú¡­¡úLn-1¡úLn, reorder it to : L0¡úLn¡úL1¡úLn - 1¡úL2¡úLn - 2¡ú¡­

  vector<int> preorderTraversal(TreeNode* root);
  //144. Binary Tree Preorder Traversal: Given a binary tree, return the preorder traversal of its nodes' values.

  vector<int> postorderTraversal(TreeNode* root);
  //145. Binary Tree Postorder Traversal: Given a binary tree, return the postorder traversal of its nodes' values.

  ListNode* insertionSortList(ListNode* head);
  //147. Insertion Sort List: Sort a linked list using insertion sort.

  ListNode* sortList(ListNode* head);
  //148. Sort List: Sort a linked list in O(n log n) time using constant space complexity.

  int evalRPN(vector<string>& tokens);
  //150. Evaluate Reverse Polish Notation: 

  void reverseWords(string &s);
  //151. Reverse Words in a String: Given an input string, reverse the string word by word.

  int maxProduct(vector<int>& nums);
  //152. Maximum Product Subarray: Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

  int findMin(vector<int>& nums);
  //153. Find Minimum in Rotated Sorted Array: Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. Find the minimum element.

  ListNode *getIntersectionNode(ListNode *headA, ListNode *headB);
  //160. Intersection of Two Linked Lists: Write a program to find the node at which the intersection of two singly linked lists begins.

  int findPeakElement(vector<int>& nums);
  //162. Find Peak Element: A peak element is an element that is greater than its neighbors.
  //Given an input array nums, where nums[i] ¡Ù nums[i + 1], find a peak element and return its index.

  int compareVersion(string version1, string version2);
  //165. Compare Version Numbers: Compare two version numbers version1 and version2. If version1 > version2 return 1; if version1 < version2 return -1;otherwise return 0.

  string fractionToDecimal(int numerator, int denominator);
  //166. Fraction to Recurring Decimal: Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
  //If the fractional part is repeating, enclose the repeating part in parentheses.

  vector<int> twoSum(vector<int>& numbers, int target);
  //167. Two Sum II - Input array is sorted: Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
  //The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.

  string convertToTitle(int n);
  //168. Excel Sheet Column Title: Given a positive integer, return its corresponding column title as appear in an Excel sheet.

  int majorityElement(vector<int>& nums);
  //169. Majority Element: Given an array of size n, find the majority element. The majority element is the element that appears more than n/2 times.
  //You may assume that the array is non - empty and the majority element always exist in the array.

  int titleToNumber(string s);
  //171. Excel Sheet Column Number: Given a column title as appear in an Excel sheet, return its corresponding column number.

  int trailingZeroes(int n);
  //172. Factorial Trailing Zeroes: Given an integer n, return the number of trailing zeroes in n!.

  void rotate(vector<int>& nums, int k);
  //189. Rotate Array: Given an array, rotate the array to the right by k steps, where k is non-negative.

  uint32_t reverseBits(uint32_t n);
  //190. Reverse Bits

  int hammingWeight(uint32_t n);
  //191. Number of 1 Bits: Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

  int rob(vector<int>& nums);
  //198. House Robber: Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
  
  vector<int> rightSideView(TreeNode* root);
  //199. Binary Tree Right Side View: Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

  int numIslands(vector<vector<char>>& grid);
  //200. Number of Islands: Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 
  //You may assume all four edges of the grid are all surrounded by water.

  int rangeBitwiseAnd(int m, int n);
  //201. Bitwise AND of Numbers Range: Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.

  bool isHappy(int n);
  //202. Happy Number: Write an algorithm to determine if a number is "happy". A happy number is a number defined by the following process : Starting with any positive integer, 
  //replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), 
  //or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

  ListNode* removeElements(ListNode* head, int val);
  //203. Remove Linked List Elements: Remove all elements from a linked list of integers that have value val.

  int countPrimes(int n);
  //204. Count Primes: Count the number of prime numbers less than a non-negative number, n.

  bool isIsomorphic(string s, string t);
  //205. Isomorphic Strings: Given two strings s and t, determine if they are isomorphic. Two strings are isomorphic if the characters in s can be replaced to get t.
  //All occurrences of a character must be replaced with another character while preserving the order of characters.No two characters may map to the same character but a character may map to itself.

  ListNode* reverseList(ListNode* head);
  //206. Reverse Linked List: Reverse a singly linked list.

  bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites);
  //207. Course Schedule: There are a total of n courses you have to take, labeled from 0 to n-1. Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair : [0, 1]
  //Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses ?

  int minSubArrayLen(int s, vector<int>& nums);
  //209. Minimum Size Subarray Sum: Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ¡Ý s. If there isn't one, return 0 instead.

  vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites);
  //210. Course Schedule II: There are a total of n courses you have to take, labeled from 0 to n-1. Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair : [0, 1]
  //Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses. There may be multiple correct orders, you just need to return one of them.If it is impossible to finish all courses, return an empty array.

  int rob2(vector<int>& nums);
  //213. House Robber II: You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle.
  //That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
  //Given a list of non - negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
  int KMP(string s, string t);//find matching position of t in s

  string shortestPalindrome(string s);
  //214. Shortest Palindrome: Given a string s, you are allowed to convert it to a palindrome by adding characters in front of it. Find and return the shortest palindrome you can find by performing this transformation.
  
  int findKthLargest(vector<int>& nums, int k);
  //215. Kth Largest Element in an Array: Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

  vector<vector<int>> combinationSum3(int k, int n);
  //216. Combination Sum III: Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.

  bool containsDuplicate(vector<int>& nums);
  //217. Contains Duplicate:Given an array of integers, find if the array contains any duplicates.¡¡Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

  bool containsNearbyDuplicate(vector<int>& nums, int k);
  //219. Contains Duplicate II: Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

  bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t);
  //220. Contains Duplicate III: Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.

  int maximalSquare(vector<vector<char>>& matrix);
  //221. Maximal Square: Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

  int countNodes(TreeNode* root);
  //222. Count Complete Tree Nodes:¡¡Given a complete binary tree, count the number of nodes.

  int computeArea(int A, int B, int C, int D, int E, int F, int G, int H);
  //223. Rectangle Area: Find the total area covered by two rectilinear rectangles in a 2D plane. Each rectangle is defined by its bottom left corner and top right corner as shown in the figure.

  TreeNode* invertTree(TreeNode* root);
  //226. Invert Binary Tree: Invert a binary tree.

  int calculate(string s);
  //227. Basic Calculator II: Implement a basic calculator to evaluate a simple expression string.
  //The expression string contains only non - negative integers, +, -, *, / operators and empty spaces.The integer division should truncate toward zero.

  vector<string> summaryRanges(vector<int>& nums);
  //228. Summary Ranges: Given a sorted integer array without duplicates, return the summary of its ranges.

  vector<int> majorityElement2(vector<int>& nums);
  //229. Majority Element II: Given an integer array of size n, find all elements that appear more than [n/3]_ times. Note: The algorithm should run in linear time and in O(1) space.

  int kthSmallest(TreeNode* root, int k);
  //230. Kth Smallest Element in a BST: Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
  
  bool isPowerOfTwo(int n);
  //231. Power of Two: Given an integer, write a function to determine if it is a power of two.

private:
  bool exploreWordSearch(int row,int col, vector<vector<bool>>& enable, int position, const vector<vector<char>>& board, const string word);
  void restoreIpDfs(string s, vector<string>& ans, int dotNum, string partStr);
  vector<TreeNode*> generateTreesDfs(int left, int right);
  void inorderBinaryTree(TreeNode* root, vector<int> &vals);
  TreeNode * buildTreeRes_preIn(vector<int>& preorder, int preStart, int preEnd, vector<int>& inorder, int inStart, int inEnd);
  TreeNode * buildTreeRes_inPos(vector<int>& inorder, int inStart, int inEnd, vector<int>& postorder, int posStart, int posEnd);
  TreeNode * buildBST(ListNode *& list, int start, int end);
  void pathSumDfs(vector<vector<int>> ans, vector<int> tmp, TreeNode* root, int sum);
  int sumNumbersRes(TreeNode* root, int curSum);
  void solveSurrondedDfs(vector<vector<char>>& board, int i, int j);
  bool isValidPali(string s, int start, int end);
  void partitionDfs(vector<vector<string>> &ans, vector<string>out, string s, int start);
  UndirectedGraphNode* cloneGraphHelper(UndirectedGraphNode *node, unordered_map<int, UndirectedGraphNode*> &umap);
  ListNode* mergeSortedList(ListNode *list1, ListNode *list2);
  void rightSVdfs(TreeNode* root, int deep, vector<int>& ans);
  void DfsNumIslands(vector<vector<char>>& grid, int row, int col);
  bool isPrime(int x, vector<int> primes);
  int robPlan(vector<int>& nums);
  void combinationSum3Dfs(int k,int n,int level, vector<int>&out,vector<vector<int>>& ans);
};
#endif // !_SOLUTION_

