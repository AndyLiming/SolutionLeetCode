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
  //143. Reorder List: Given a singly linked list L: L0→L1→…→Ln-1→Ln, reorder it to : L0→Ln→L1→Ln - 1→L2→Ln - 2→…

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
  //Given an input array nums, where nums[i] ≠ nums[i + 1], find a peak element and return its index.

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
  bool canFinish(int numCourses, vector<vector<int>>& prerequisites);
  //207. Course Schedule: There are a total of n courses you have to take, labeled from 0 to n-1. Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair : [0, 1]
  //Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses ?

  int minSubArrayLen(int s, vector<int>& nums);
  //209. Minimum Size Subarray Sum: Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

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
  //217. Contains Duplicate:Given an array of integers, find if the array contains any duplicates.　Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.

  bool containsNearbyDuplicate(vector<int>& nums, int k);
  //219. Contains Duplicate II: Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

  bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t);
  //220. Contains Duplicate III: Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.

  int maximalSquare(vector<vector<char>>& matrix);
  //221. Maximal Square: Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

  int countNodes(TreeNode* root);
  //222. Count Complete Tree Nodes:　Given a complete binary tree, count the number of nodes.

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

  bool isPalindrome(ListNode* head);
  //234. Palindrome Linked List: Given a singly linked list, determine if it is a palindrome.

  TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
  //235. Lowest Common Ancestor of a Binary Search Tree:　Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

  TreeNode* lowestCommonAncestor2(TreeNode* root, TreeNode* p, TreeNode* q);
  //236. Lowest Common Ancestor of a Binary Tree: Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

  void deleteNode(ListNode* node);
  //237. Delete Node in a Linked List: Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.

  vector<int> productExceptSelf(vector<int>& nums);
  //238. Product of Array Except Self: Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

  bool searchMatrix2(vector<vector<int>>& matrix, int target);
  //240. Search a 2D Matrix II: Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
  //Integers in each row are sorted in ascending from left to right. Integers in each column are sorted in ascending from top to bottom.

  vector<int> diffWaysToCompute(string input);
  //241. Different Ways to Add Parentheses: Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. The valid operators are +, - and *.

  bool isAnagram(string s, string t);
  //242. Valid Anagram: Given two strings s and t , write a function to determine if t is an anagram of s.

  vector<string> binaryTreePaths(TreeNode* root);
  //257. Binary Tree Paths: Given a binary tree, return all root-to-leaf paths.

  int addDigits(int num);
  //258. Add Digits: Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

  vector<int> singleNumber3(vector<int>& nums);
  //260. Single Number III: Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.

  bool isUgly(int num);
  //263. Ugly Number: Write a program to check whether a given number is an ugly number. Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.

  int nthUglyNumber(int n);
  //264. Ugly Number II: Write a program to find the n-th ugly number. Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.

  int missingNumber(vector<int>& nums);
  //268. Missing Number: Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

  bool isAdditiveNumber(string num);
  //306. Additive Number: Additive number is a string whose digits can form additive sequence. A valid additive sequence should contain at least three numbers.Except for the first two numbers, each subsequent number in the sequence must be the sum of the preceding two.
  //Given a string containing only digits '0' - '9', write a function to determine if it's an additive number.

  int nthSuperUglyNumber(int n, vector<int>& primes);
  //313. Super Ugly Number: Write a program to find the nth super ugly number.
  //Super ugly numbers are positive numbers whose all prime factors are in the given prime list primes of size k.

  bool isPowerOfThree(int n);
  //326. Power of Three: Given an integer, write a function to determine if it is a power of three.

  int maxProductWordLength(vector<string>& words);
  //318. Maximum Product of Word Lengths: Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. 
  //You may assume that each word will contain only lower case letters. If no such two words exist, return 0.

  ListNode* oddEvenList(ListNode* head);
  //328. Odd Even Linked List: Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.
  //You should try to do it in place.The program should run in O(1) space complexity and O(nodes) time complexity.

  int bulbSwitch(int n);
  //319. Bulb Switcher: There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on).
  //For the i-th round, you toggle every i bulb. For the n-th round, you only toggle the last bulb. Find how many bulbs are on after n rounds.

  int rob3(TreeNode* root);
  //337. House Robber III: all houses in this place forms a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.
  //Determine the maximum amount of money the thief can rob tonight without alerting the police.

  int coinChange(vector<int>& coins, int amount);
  //322. Coin Change: You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount.
  //If that amount of money cannot be made up by any combination of the coins, return -1.

  bool isPowerOfFour(int num);
  //342. Power of Four: Given an integer (signed 32 bits), write a function to check whether it is a power of 4.

  vector<int> countBits(int num);
  //338. Counting Bits: Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

  int maxProfit(vector<int>& prices);
  //309. Best Time to Buy and Sell Stock with Cooldown: Say you have an array for which the ith element is the price of a given stock on day i.
  //Design an algorithm to find the maximum profit.You may complete as many transactions as you like(ie, buy one and sell one share of the stock multiple times) with the following restrictions :

  int countRangeSum(vector<int>& nums, int lower, int upper);
  //327. Count of Range Sum: Given an integer array nums, return the number of range sums that lie in [lower, upper] inclusive.
  //Range sum S(i, j) is defined as the sum of the elements in nums between indices i and j(i ≤ j), inclusive.

  int integerBreak(int n);
  //343. Integer Break: Given a positive integer n, break it into the sum of at least two positive integers and maximize the product of those integers. Return the maximum product you can get.

  void wiggleSort(vector<int>& nums);
  //324. Wiggle Sort II: Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....

  void reverseString(vector<char>& s);
  //344. Reverse String

  string reverseVowels(string s);
  //345. Reverse Vowels of a String

  vector<int> topKFrequent(vector<int>& nums, int k);
  //347. Top K Frequent Elements: Given a non-empty array of integers, return the k most frequent elements.

  vector<int> intersection(vector<int>& nums1, vector<int>& nums2);
  //349. Intersection of Two Arrays: Given two arrays, write a function to compute their intersection.

  vector<int> intersect(vector<int>& nums1, vector<int>& nums2);
  //350. Intersection of Two Arrays II

  bool increasingTriplet(vector<int>& nums);
  //334. Increasing Triplet Subsequence: Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array

  int maxEnvelopes(vector<pair<int, int>>& envelopes);
  //354. Russian Doll Envelopes: You have a number of envelopes with widths and heights given as a pair of integers (w, h). One envelope can fit into another if and only if both the width and height of one envelope is greater than the width and height of the other envelope.
  //What is the maximum number of envelopes can you Russian doll ? (put one inside other)

  int countNumbersWithUniqueDigits(int n);
  //357. Count Numbers with Unique Digits: Given a non-negative integer n, count all numbers with unique digits, x, where 0 ≤ x < 10n.

  int maxSumSubmatrix(vector<vector<int>>& matrix, int k);
  //363. Max Sum of Rectangle No Larger Than K: Given a non-empty 2D matrix matrix and an integer k, find the max sum of a rectangle in the matrix such that its sum is no larger than k.

  bool canMeasureWater(int x, int y, int z);
  //365. Water and Jug Problem: You are given two jugs with capacities x and y litres. There is an infinite amount of water supply available. You need to determine whether it is possible to measure exactly z litres using these two jugs.
  //If z liters of water is measurable, you must have z liters of water contained within one or both buckets by the end.

  bool isPerfectSquare(int num);
  //367. Valid Perfect Square: Given a positive integer num, write a function which returns True if num is a perfect square else False.

  vector<int> largestDivisibleSubset(vector<int>& nums);
  //368. Largest Divisible Subset: Given a set of distinct positive integers, find the largest subset such that every pair (Si, Sj) of elements in this subset satisfies:
  //Si % Sj = 0 or Sj % Si = 0. If there are multiple solutions, return any subset is fine.

  int getSum(int a, int b);
  //371. Sum of Two Integers: Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.

  int superPow(int a, vector<int>& b);
  //372. Super Pow: Your task is to calculate ab mod 1337 where a is a positive integer and b is an extremely large positive integer given in the form of an array.

  vector<pair<int, int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k);
  //373. Find K Pairs with Smallest Sums: You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.
  //Define a pair(u, v) which consists of one element from the first array and one element from the second array.
  //Find the k pairs(u1, v1), (u2, v2) ...(uk, vk) with the smallest sums.

  int getMoneyAmount(int n);
  //375. Guess Number Higher or Lower II: We are playing the Guess Game. The game is as follows: I pick a number from 1 to n.You have to guess which number I picked.
  //Every time you guess wrong, I'll tell you whether the number I picked is higher or lower. However, when you guess a particular number x, and you guess wrong, you pay $x.You win the game when you guess the number I picked.

  int wiggleMaxLength(vector<int>& nums);
  //376. Wiggle Subsequence:A sequence of numbers is called a wiggle sequence if the differences between successive numbers strictly alternate between positive and negative. The first difference (if one exists) may be either positive or negative. 
  //A sequence with fewer than two elements is trivially a wiggle sequence.

  int combinationSum4(vector<int>& nums, int target);
  //377. Combination Sum IV: Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.

  int kthSmallest(vector<vector<int>>& matrix, int k);
  //378. Kth Smallest Element in a Sorted Matrix: Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.
  //Note that it is the kth smallest element in the sorted order, not the kth distinct element.

  bool canConstruct(string ransomNote, string magazine);
  //383. Ransom Note: Given an arbitrary ransom note string and another string containing letters from all the magazines, write a function that will return true if the ransom note can be constructed from the magazines ; otherwise, it will return false. 
  //Each letter in the magazine string can only be used once in your ransom note.

  vector<int> lexicalOrder(int n);
  //386. Lexicographical Numbers: Given an integer n, return 1 - n in lexicographical order. //For example, given 13, return: [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9].
  //Please optimize your algorithm to use less time and space.The input size may be as large as 5, 000, 000.

  int firstUniqChar(string s);
  //387. First Unique Character in a String: Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1. 

  char findTheDifference(string s, string t);
  //389. Find the Difference: Given two strings s and t which consist of only lowercase letters. String t is generated by random shuffling string s and then add one more letter at a random position.
  //Find the letter that was added in t.

  int minRefuelStops(int target, int startFuel, vector<vector<int>>& stations);
  //871. Minimum Number of Refueling Stops: A car travels from a starting position to a destination which is target miles east of the starting position.
  // Along the way, there are gas stations.Each station[i] represents a gas station that is station[i][0] miles east of the starting position, and has station[i][1] liters of gas.
  //The car starts with an infinite tank of gas, which initially has startFuel liters of fuel in it.It uses 1 liter of gas per 1 mile that it drives.
  //When the car reaches a gas station, it may stop and refuel, transferring all the gas from the station into the car.
  //What is the least number of refueling stops the car must make in order to reach its destination ? If it cannot reach the destination, return -1.

  int lastRemaining(int n);
  //390. Elimination Game: There is a list of sorted integers from 1 to n. Starting from left to right, remove the first number and every other number afterward until you reach the end of the list.
  //Repeat the previous step again, but this time from right to left, remove the right most number and every other number from the remaining numbers.
  //We keep repeating the steps again, alternating left to right and right to left, until a single number remains. Find the last number that remains starting with a list of length n.

  bool isSubsequence(string s, string t);
  //Given a string s and a string t, check if s is subsequence of t.  You may assume that there is only lower case English letters in both s and t.t is potentially a very long(length ~= 500, 000) string, and s is a short string(<= 100).

  string decodeString(string s);
  //394. Decode String: Given an encoded string, return it's decoded string. The encoding rule is : k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times.Note that k is guaranteed to be a positive integer.
  //You may assume that the input string is always valid; No extra white spaces, square brackets are well - formed, etc.
  //Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k.For example, there won't be input like 3a or 2[4]. 

  int longestSubstring(string s, int k);
  //395. Longest Substring with At Least K Repeating Characters: Find the length of the longest substring T of a given string (consists of lowercase letters only) such that every character in T appears no less than k times. 

  bool validUtf8(vector<int>& data);
  //393. UTF-8 Validation: A character in UTF8 can be from 1 to 4 bytes long, subjected to the following rules:
  //For 1 - byte character, the first bit is a 0, followed by its unicode code. For n - bytes character, the first n - bits are all one's, the n+1 bit is 0, followed by n-1 bytes with most significant 2 bits being 10.
  //Given an array of integers representing the data, return whether it is a valid utf-8 encoding. 

  int maxRotateFunction(vector<int>& A);
  //396. Rotate Function: iven an array of integers A and let n to be its length. 
  //Assume Bk to be an array obtained by rotating the array A k positions clock - wise, we define a "rotation function" F on A as follow :
  //F(k) = 0 * Bk[0] + 1 * Bk[1] + ... + (n - 1) * Bk[n - 1]. Calculate the maximum value of F(0), F(1), ..., F(n - 1).

  int integerReplacement(int n);
  //397. Integer Replacement: Given a positive integer n and you can do operations as follow: If n is even, replace n with n / 2. If n is odd, you can replace n with either n + 1 or n - 1.
  //What is the minimum number of replacements needed for n to become 1 ?

  int findNthDigit(int n);
  //400. Nth Digit: Find the nth digit of the infinite integer sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... 

  vector<double> calcEquation(vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries);
  //399. Evaluate Division: Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number(floating point number).
  //Given some queries, return the answers.If the answer does not exist, return -1.0.

  int sumOfLeftLeaves(TreeNode* root);
  //404. Sum of Left Leaves: Find the sum of all left leaves in a given binary tree.

  ListNode* reverseKGroup(ListNode* head, int k);
  //25. Reverse Nodes in k-Group: Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
  //k is a positive integer and is less than or equal to the length of the linked list.If the number of nodes is not a multiple of k then left - out nodes in the end should remain as it is.

  ListNode* mergeKLists(vector<ListNode*>& lists);
  //23. Merge k Sorted Lists: Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

  int longestValidParentheses(string s);
  //32. Longest Valid Parentheses

  vector<vector<int>> reconstructQueue(vector<vector<int>>& people);
  //406. Queue Reconstruction by Height: uppose you have a random list of people standing in a queue. Each person is described by a pair of integers (h, k), 
  //where h is the height of the person and k is the number of people in front of this person who have a height greater than or equal to h. Write an algorithm to reconstruct the queue. 

  bool canCross(vector<int>& stones);
  //403. Frog Jump: A frog is crossing a river. The river is divided into x units and at each unit there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.
  //Given a list of stones' positions (in units) in sorted ascending order, determine if the frog is able to cross the river by landing on the last stone. Initially, the frog is on the first stone and assume the first jump must be 1 unit.
  //If the frog's last jump was k units, then its next jump must be either k - 1, k, or k + 1 units. Note that the frog can only jump in the forward direction.

  int longestPalindrome(string s);
  //409. Longest Palindrome: Given a string which consists of lowercase or uppercase letters, find the length of the longest palindromes that can be built with those letters.
  //This is case sensitive, for example "Aa" is not considered a palindrome here.

  vector<string> fizzBuzz(int n);
  //412. Fizz Buzz: Write a program that outputs the string representation of numbers from 1 to n.
  //But for multiples of three it should output “Fizz” instead of the number and for the multiples of five output “Buzz”.
  //For numbers which are multiples of both three and five output “FizzBuzz”.

  int splitArray(vector<int>& nums, int m);
  //410. Split Array Largest Sum:
  //Given an array which consists of non - negative integers and an integer m, you can split the array into m non - empty continuous subarrays.
  //Write an algorithm to minimize the largest sum among these m subarrays.

  int minDistance(string word1, string word2);
  //72. Edit Distance: Given two words word1 and word2, find the minimum number of operations required to convert word1 to word2.

  vector<int> maxSlidingWindow(vector<int>& nums, int k);
  //239. Sliding Window Maximum: Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. 
  //Each time the sliding window moves right by one position. Return the max sliding window.

  string serialize(TreeNode* root);
  TreeNode* deserialize(string data);
  //297. Serialize and Deserialize Binary Tree

  vector<vector<int>> insertIntervals(vector<vector<int>>& intervals, vector<int>& newInterval);
  //57. Insert Interval: Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
  //You may assume that the intervals were initially sorted according to their start times.

  int minPatches(vector<int>& nums, int n);
  //330. Patching Array: Given a sorted positive integer array nums and an integer n, add/patch elements to the array such that any number in range [1, n] inclusive can be formed by the sum of some elements in the array.
  //Return the minimum number of patches required.

  int firstMissingPositive(vector<int>& nums);
  //41. First Missing Positive: Given an unsorted integer array, find the smallest missing positive integer.

  int trapRainWater(vector<int>& height);
  //42. Trapping Rain Water: Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

  int countDigitOne(int n);
  //233. Number of Digit One: Given an integer n, count the total number of digit 1 appearing in all non-negative integers less than or equal to n.

  vector<int> findAnagrams(string s, string p);
  //438. Find All Anagrams in a String: Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
  //Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20, 100. The order of output does not matter.

  int findTargetSumWays(vector<int>& nums, int S);
  //494. Target Sum: You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.
  //Find out how many ways to assign symbols to make sum of integers equal to target S.

  vector<int> dailyTemperatures(vector<int>& T);
  //739. Daily Temperatures: Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. 
  //If there is no future day for which this is possible, put 0 instead.

  string minWindow(string s, string t);
  //76. Minimum Window Substring: Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

  vector<string> wordBreak2(string s, vector<string>& wordDict);
  //140. Word Break II: Given a non-empty string s and a dictionary wordDict containing a list of non-empty words,
  //add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences.

  int maxPathSum(TreeNode* root);
  //124. Binary Tree Maximum Path Sum: Given a non-empty binary tree, find the maximum path sum.
  //For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent - child connections.
  //The path must contain at least one node and does not need to go through the root.

  TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2);
  //617. Merge Two Binary Trees: Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.
  //You need to merge them into a new binary tree.The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node.Otherwise, the NOT null node will be used as the node of new tree.

  int longestConsecutive(vector<int>& nums);
  //128. Longest Consecutive Sequence: Given an unsorted array of integers, find the length of the longest consecutive elements sequence. time complexity O(n)

  int pathSum3(TreeNode* root, int sum);
  //437. Path Sum III: You are given a binary tree in which each node contains an integer value. Find the number of paths that sum to a given value.
  //The path does not need to start or end at the root or a leaf, but it must go downwards(traveling only from parent nodes to child nodes).
  //The tree has no more than 1, 000 nodes and the values are in the range - 1, 000, 000 to 1, 000, 000.

  int findUnsortedSubarray(vector<int>& nums);
  //581. Shortest Unsorted Continuous Subarray: Given an integer array, you need to find one continuous subarray that if you only sort this subarray in ascending order, 
  //then the whole array will be sorted in ascending order, too.
  //You need to find the shortest such subarray and output its length.

  int countSubstrings(string s);
  //647. Palindromic Substrings: Given a string, your task is to count how many palindromic substrings in this string.
  //The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

  int longestIncreasingPath(vector<vector<int>>& matrix);
  //329. Longest Increasing Path in a Matrix: Given an integer matrix, find the length of the longest increasing path.
  //From each cell, you can either move to four directions : left, right, up or down.You may NOT move diagonally or move outside of the boundary(i.e.wrap - around is not allowed).

  vector<vector<string>> solveNQueens(int n);
  //51. N-Queens: The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.

  int diameterOfBinaryTree(TreeNode* root);
  //543. Diameter of Binary Tree: Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

  int leastInterval(vector<char>& tasks, int n);
  //621. Task Scheduler: Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. 
  //Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.
  //However, there is a non - negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.
  //You need to return the least number of intervals the CPU will take to finish all the given tasks.

  vector<int> findDisappearedNumbers(vector<int>& nums);
  //448. Find All Numbers Disappeared in an Array: Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.
  //Find all the elements of[1, n] inclusive that do not appear in this array.
  //Could you do it without extra space and in O(n) runtime ? You may assume the returned list does not count as extra space.

  vector<string> removeInvalidParentheses(string s);
  //301. Remove Invalid Parentheses: Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

  int maxCoins(vector<int>& nums);
  //312. Burst Balloons: Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number on it represented by array nums. 
  //You are asked to burst all the balloons. If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins. 
  //Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.
  //Find the maximum coins you can collect by bursting the balloons wisely.

  vector<vector<int>> levelOrder(MultiTreeNode* root);
  //429. N-ary Tree Level Order Traversal: Given an n-ary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

  vector<int> findDuplicates(vector<int>& nums);
  //442. Find All Duplicates in an Array: Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.
  //Find all the elements that appear twice in this array.
  //Could you do it without extra spaceand in O(n) runtime ?

  void gameOfLife(vector<vector<int>>& board);
  //289. Game of Life

  vector<int> findSubstring(string s, vector<string>& words);
  //30. Substring with Concatenation of All Words: You are given a string, s, and a list of words, words, that are all of the same length. 
  //Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.

  int numDistinct(string s, string t);
  //115. Distinct Subsequences
  //Given a string S and a string T, count the number of distinct subsequences of S which equals T.
  //A subsequence of a string is a new string which is formed from the original string by deleting some(can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).
  //It's guaranteed the answer fits on a 32-bit signed integer.

  int maxPoints(vector<vector<int>>& points);
  //149. Max Points on a Line
  //Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.

  int candy(vector<int>& ratings);
  //135. Candy
  //There are N children standing in a line. Each child is assigned a rating value. You are giving candies to these children subjected to the following requirements :
  //  Each child must have at least one candy. Children with a higher rating get more candies than their neighbors. What is the minimum candies you must give ?

  vector<vector<int>> pacificAtlantic(vector<vector<int>>& matrix);
  
  //417. Pacific Atlantic Water Flow
  //Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and bottom edges.
  //Water can only flow in four directions(up, down, left, or right) from a cell to another one with height equal or lower.
  //Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.

  string validIPAddress(string IP);
  //468. Validate IP Address: Write a function to check whether an input string is a valid IPv4 address or IPv6 address or neither.

  int hIndex2(vector<int>& citations);
  //275. H-Index II: Given an array of citations sorted in ascending order (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.

  bool isPalindromeString(string s);
  //125

  string longestDupSubstring(string S);
  //1044.

  bool isMatch(string s, string p);
  //10. Regular Expression Matching: Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
  //'.' Matches any single character.
  //'*' Matches zero or more of the preceding element.
  //The matching should cover the entire input string(not partial).

  int threeSumClosest(vector<int>& nums, int target);
  //Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. 
  //Return the sum of the three integers. You may assume that each input would have exactly one solution.

  vector<string> findItinerary(vector<vector<string>>& tickets);
  //332. Reconstruct Itinerary: Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], 
  //reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.

  int findLength(vector<int>& A, vector<int>& B);
  //718. Maximum Length of Repeated Subarray: Given two integer arrays A and B, return the maximum length of an subarray that appears in both arrays.

  int arrangeCoins(int n);
  //441. Arranging Coins:You have a total of n coins that you want to form in a staircase shape, where every k-th row must have exactly k coins.
  //Given n, find the total number of full staircase rows that can be formed.
  //n is a non - negative integer and fits within the range of a 32 - bit signed integer.

  bool isInterleave(string s1, string s2, string s3);
  //97. Interleaving String: Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.

  vector<vector<int>> levelOrderBottom(TreeNode* root);
  //107. Binary Tree Level Order Traversal II: Given a binary tree, return the bottom-up level order traversal of its nodes' values. 
  //(ie, from left to right, level by level from leaf to root).

  double new21Game(int N, int K, int W);
  //837. New 21 Game: Alice plays the following game, loosely based on the card game "21".
  //Alice starts with 0 points, and draws numbers while she has less than K points.During each draw, she gains an integer number of points randomly from the range[1, W], 
  //where W is an integer.Each draw is independentand the outcomes have equal probabilities.
  //Alice stops drawing numbers when she gets K or more points.What is the probability that she has N or less points ?

  TreeNode* sortedArrayToBST(vector<int>& nums);
  //108. Convert Sorted Array to Binary Search Tree: Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
  //For this problem, a height - balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

  vector<int> prisonAfterNDays(vector<int>& cells, int N);
  //957. Prison Cells After N Days: There are 8 prison cells in a row, and each cell is either occupied or vacant.
  //Each day, whether the cell is occupied or vacant changes according to the following rules :
  //If a cell has two adjacent neighbors that are both occupied or both vacant, then the cell becomes occupied.
  //Otherwise, it becomes vacant. (Note that because the prison is a row, the firstand the last cells in the row can't have two adjacent neighbors.)

  int islandPerimeter(vector<vector<int>>& grid);
  //463. Island Perimeter

  vector<vector<int>> threeSum(vector<int>& nums);
  //3Sum: Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
  //The solution set must not contain duplicate triplets.
  
  int widthOfBinaryTree(TreeNode* root);
  //662. Maximum Width of Binary Tree: Given a binary tree, write a function to get the maximum width of the given tree. The width of a tree is the maximum width among all levels. 
  //The binary tree has the same structure as a full binary tree, but some nodes are null.
  //The width of one level is defined as the length between the end - nodes(the leftmost and right most non - null nodes in the level, where the null nodes between the end - nodes are also counted into the length calculation.

  Node* flatten(Node* head);
  //430. Flatten a Multilevel Doubly Linked List

  bool isBipartite(vector<vector<int>>& graph);
  //785 
  void recoverTree(TreeNode* root);
  //99. Recover Binary Search Tree: Two elements of a binary search tree (BST) are swapped by mistake. Recover the tree without changing its structure.

  bool divisorGame(int N);
  //1025. Divisor Game: Alice and Bob take turns playing a game, with Alice starting first. Initially, there is a number N on the chalkboard.On each player's turn, that player makes a move consisting of:
  //Choosing any x with 0 < x < N and N % x == 0. Replacing the number N on the chalkboard with N - x. Also, if a player cannot make a move, they lose the game.
  //Return True ifand only if Alice wins the game, assuming both players play optimally.
  
  vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph);
  //797.All Paths From Source to Target:Given a directed, acyclic graph of N nodes.  Find all possible paths from node 0 to node N-1, and return them in any order.
  //The graph is given as follows : the nodes are 0, 1, ..., graph.length - 1.  graph[i] is a list of all nodes j for which the edge(i, j) exists.

  int getLengthOfOptimalCompression(string s, int k);
  //1531. String Compression II: Run-length encoding is a string compression method that works by replacing consecutive identical characters (repeated 2 or more times) with the concatenation of the character and the number marking the count of the characters (length of the run). 
  //For example, to compress the string "aabccc" we replace "aa" by "a2" and replace "ccc" by "c3". Thus the compressed string becomes "a2bc3".
  //Notice that in this problem, we are not adding '1' after single characters.
  //Given a string s and an integer k.You need to delete at most k characters from s such that the run - length encoded version of s has minimum length.
  //Find the minimum length of the run - length encoded version of s after deleting at most k characters.
  
  vector<int> smallestRange(vector<vector<int>>& nums);
  //632. Smallest Range Covering Elements from K Lists: You have k lists of sorted integers in ascending order. Find the smallest range that includes at least one number from each of the k lists.
  //We define the range[a, b] is smaller than range[c, d] if b - a < d - c or a < c if b - a == d - c.

  bool detectCapitalUse(string word);
  //520. Detect Capital: Given a word, you need to judge whether the usage of capitals in it is right or not.
  //We define the usage of capitals in a word to be right when one of the following cases holds :
  //All letters in this word are capitals, like "USA".
  //All letters in this word are not capitals, like "leetcode".
  //Only the first letter in this word is capital, like "Google".
  //Otherwise, we define that this word doesn't use capitals in a right way.

  string addStrings(string num1, string num2);
  //415. Add Strings: Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.
  //Note:
  //The length of both num1and num2 is < 5100.
  //Both num1 and num2 contains only digits 0 - 9.
  //Both num1 and num2 does not contain any leading zero.
  //You must not use any built - in BigInteger library or convert the inputs to integer directly.

  vector<vector<int>> palindromePairs(vector<string>& words);
  //336. Palindrome Pairs: Given a list of unique words, find all pairs of distinct indices (i, j) in the given list, 
  //so that the concatenation of the two words, i.e. words[i] + words[j] is a palindrome.

  int countBinarySubstrings(string s);
  //696

  int orangesRotting(vector<vector<int>>& grid);
  //994. Rotting Oranges: In a given grid, each cell can have one of three values:
  //the value 0 representing an empty cell;
  //the value 1 representing a fresh orange;
  //the value 2 representing a rotten orange.
  //Every minute, any fresh orange that is adjacent(4 - directionally) to a rotten orange becomes rotten.
  //Return the minimum number of minutes that must elapse until no cell has a fresh orange.If this is impossible, return -1 instead.

  int maxProfit3(vector<int>& prices);
  //123. Best Time to Buy and Sell Stock III

  vector<int> distributeCandies(int candies, int num_people);
  //1103. Distribute Candies to People

  string toGoatLatin(string S);
  //824. Goat Latin: 

  vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click);
  //529. Minesweeper

  bool judgePoint24(vector<int>& nums);
  //679. 24 game

  vector<string> letterCombinations(string digits);
  //17

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
  void serchBtPaths(TreeNode* root, vector<string> &ans, string out);
  bool additiveNumBacktrack(long long num1, long long num2, string s, int start);
  vector<int>rob3Dfs(TreeNode* root);
  bool comparePair(pair<int, int>& p1, pair<int, int>& p2);
  int powMod(int a, int k,int base);
  int moneyAcountHelper(int start, int end, vector<vector<int>> &dp);
  int helpLastRemaining(int n, bool l2r);
  double calEquhelper(string up, string down, unordered_set<string>& visited, unordered_map<string, unordered_map<string, double>> &m);
  void sumLeftLeavesCore(TreeNode*root, int& sum);
  ListNode* reverseList(ListNode * start, ListNode * end);
  ListNode* mergeTwoList(ListNode* h1, ListNode * h2);
  bool cansplit(vector<int>& nums, int value, int m);
  vector<string> wordBreak2Rec(string s, vector<string>& wordDict, unordered_map<string, vector<string>>&wordMap);
  int maxPathSumRec(TreeNode* node, int& res);
  void pathSum3Dfs(TreeNode * node,int curSum,int target,int &nums, vector<TreeNode*>& out);
  int longestIncPathDfs(vector<vector<int>>& matrix, int row, int col, vector<vector<int>>&dirs, vector<vector<int>>&dp);
  bool nQueensIsValid(int n, int k, vector<int>&colIds);
  void nQueensCore(int n, int k, vector<int>&colIds, vector<vector<string>>&ans);
  int diameterOfBinaryTreeCore(TreeNode* root,int & maxDia);
  void removeInvPaCore(string s, int lastI, int lastJ, char parentheses[], vector<string>&ans);
  void pacificAtlanticDFS(vector<vector<int>>& matrix, int x, int y, vector<vector<bool>>& visit, int pre);
  bool ifIpv4(string ip);
  bool ifIpv6(string ip);
  void findItineraryDFS(string cur, vector<string>& ans, unordered_map < string, priority_queue<string, vector<string>, greater<string>>>& table);
  TreeNode* sortedArrayToBSTCore(vector<int>& nums, int l, int r);
  vector<int> prisonAfterNDaysNext(vector<int>& cells);
  bool isBipartiteDfs(vector<vector<int>>& graph, vector<int>& color, int v, int c);
  //No 99
  void recoverBSTinorder(TreeNode* root);
  TreeNode* pre = nullptr, * first = nullptr, * second = nullptr;
  /****/
  void allPathsSourceTargetDFS(vector<vector<int>>& graph,int n,int k,vector<int>&cur,vector<vector<int>>&ans);

  //No 336
  vector<string> wordsRev;
  unordered_map<string, int> indices;
  int findWord336(const string& s, int left, int right);
  bool isPalindrome336(const string& s, int left, int right);
  /****/

  void updateBoardDfs(vector<vector<char>>& board, int x, int y, vector<int>& dirX, vector<int>& dirY);
  //No.529 Minesweeper

  bool recJudgePoint24(vector<double>& nums);
  //No 679
  void backtraceLetterComb(vector<string>& ans, string cur, string digits, int id, unordered_map<char, string>& m);

  string reverseWords(string s);
  //No 557

  bool canVisitAllRooms(vector<vector<int>>& rooms);
  //No 841

  bool PredictTheWinner(vector<int>& nums);
  //486.

  bool wordPattern(string pattern, string str);
  //290.Word Pattern

  string getHint(string secret, string guess);
  //299.Bulls and Cows
};

//No 303 Range Sum Query - Immutable
class NumArray {
public:
  NumArray(vector<int> nums) {
    sumDp = nums;
    for (int i = 1;i < nums.size();++i) {
      sumDp[i] += sumDp[i - 1];
    }
  }

  int sumRange(int i, int j) {
    if (i == 0) return sumDp[j];
    else return sumDp[j] - sumDp[i - 1];
  }
private:
  vector<int>sumDp;
};

//No 304 Range Sum Query 2D - Immutable
class NumMatrix {
public:
  NumMatrix(vector<vector<int>> matrix) {
    if (matrix.empty() || matrix[0].empty()) return;
    sumDp.resize(matrix.size() + 1, vector<int>(matrix[0].size() + 1, 0));
    for (int i = 1;i <= matrix.size();++i) {
      for (int j = 1;j <= matrix[0].size();++j) {
        sumDp[i][j] = sumDp[i][j - 1] + sumDp[i - 1][j] - sumDp[i - 1][j - 1]+ matrix[i-1][j-1];
      }
    }
  }

  int sumRegion(int row1, int col1, int row2, int col2) {
    return sumDp[row2 + 1][col2 + 1] - sumDp[row1][col2 + 1] - sumDp[row2 + 1][col1] + sumDp[row1][col1];
  }
private :
  vector<vector<int>> sumDp;
};

//No 295 Find Median from Data Stream
class MedianFinder {
public:
  /** initialize your data structure here. */
  MedianFinder():size(0) {
 
  }

  void addNum(int num) {
    ++size;
    l.push(num);
    g.push(l.top());
    l.pop();
    if (l.size() < g.size()) {
      l.push(g.top());
      g.pop();
    }
  }

  double findMedian() {
    if (size % 2 == 1) return l.top();
    else return (l.top() + g.top()) / 2;
  }
private:
  int size;
  priority_queue<int, vector<int>, greater<int>> g;
  priority_queue<int, vector<int>, less<int>> l;
};
//No 380 Insert Delete GetRandom O(1)
class RandomizedSet {
public:
  /** Initialize your data structure here. */
  RandomizedSet() {

  }

  /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
  bool insert(int val) {
    if (hashMap.find(val) != hashMap.end()) return false;
    else {
      hashMap.insert({ val,elemNum.size() });
      elemNum.push_back(val);
      return true;
    }
  }

  /** Removes a value from the set. Returns true if the set contained the specified element. */
  bool remove(int val) {
    if (hashMap.find(val) == hashMap.end()) return false;
    else {
      auto p = hashMap.find(val);
      int index = p->second;
      swap(elemNum[index], elemNum[elemNum.size() - 1]);
      elemNum.pop_back();
      hashMap[elemNum[index]] = index;
      hashMap.erase(val);
      return true;
    }
  }

  /** Get a random element from the set. */
  int getRandom() {  
    if (elemNum.size() == 0)return -1;
    else {
      uniform_int_distribution<int>distri(0, elemNum.size() - 1);
      return elemNum[distri(generator)];
    }
  }
private:
  unordered_map<int, int> hashMap;
  vector<int>elemNum;
  default_random_engine generator;
};

//No 146 LRU Cache
class LRUCache {
public:
  LRUCache(int capacity) {
    cap = capacity;
  }

  int get(int key) {
    auto it = m.find(key);
    if (it == m.end()) return -1;
    l.splice(l.begin(), l, it->second);
    return it->second->second;
  }

  void put(int key, int value) {
    auto it = m.find(key);
    if (it != m.end()) l.erase(it->second);
    l.push_front({ key,value });
    m[key] = l.begin();
    if (m.size() > cap) {
      int k = l.rbegin()->first;
      l.pop_back();
      m.erase(k);
    }
  }
private:
  int cap;
  unordered_map<int, list<pair<int, int>>::iterator> m;
  list<pair<int, int>>l;
};

class MinStack {
public:
  /** initialize your data structure here. */
  MinStack() {
    size = 0;
  }
  void push(int x) {
    data.push(x);
    if (minD.empty()) {
      minD.push(x);
    }
    else {
      int curMin = minD.top();
      minD.push(min(curMin, x));
    }
  }

  void pop() {
    if (!data.empty()) {
      data.pop();
      minD.pop();
    }
  }

  int top() {
    if (data.empty()) return -1;
    return data.top();
  }

  int getMin() {
    if (minD.empty()) return -1;
    return minD.top();
  }
  bool isEmpty() {
    return data.empty();
  }
private:
  stack<int> data, minD;
  int size;
};

//No 208 Implement Trie (Prefix Tree)
class Trie {
public:
  class TrieNode {
  public:
    TrieNode *children[26];
    bool isword;
    TrieNode(): isword(false) {
      for (auto &c : children) c = nullptr;
    }
  };
  /** Initialize your data structure here. */
  Trie() {
    root = new TrieNode();
  }

  /** Inserts a word into the trie. */
  void insert(string word) {
    TrieNode *p = root;
    for (auto w : word) {
      int i = w - 'a';
      if (!p->children[i])p->children[i] = new TrieNode();
      p = p->children[i];
    }
    p->isword = true;
  }

  /** Returns if the word is in the trie. */
  bool search(string word) {
    TrieNode *p = root;
    for (auto w : word) {
      int i = w - 'a';
      if (!p->children[i]) return false;
      p = p->children[i];
    }
    return p->isword;
  }

  /** Returns if there is any word in the trie that starts with the given prefix. */
  bool startsWith(string prefix) {
    TrieNode *p = root;
    for (auto w : prefix) {
      int i = w - 'a';
      if (!p->children[i]) return false;
      p = p->children[i];
    }
    return true;
  }
private:
  TrieNode *root;
};

//705. Design HashSet 
//Design a HashSet without using any built - in hash table libraries.
//To be specific, your design should include these functions :
//add(value) : Insert a value into the HashSet.
//contains(value) : Return whether the value exists in the HashSet or not.
//remove(value) : Remove a value in the HashSet.If the value does not exist in the HashSet, do nothing.
class MyHashSet {
public:
  /** Initialize your data structure here. */
  MyHashSet() {
    bs.reset();
  }

  void add(int key) {
    if (bs[key] == 0) bs.set(key);
  }

  void remove(int key) {
    if (bs[key] == 1)bs.reset(key);
  }

  /** Returns true if this set contains the specified element */
  bool contains(int key) {
    return bs[key]==1;
  }
private:
  bitset<1000000>bs;
};

//LCP 13 xun bao
//https://leetcode-cn.com/problems/xun-bao/
class SolutionLCP13 {
public:

  int dx[4] = { 1, -1, 0, 0 };
  int dy[4] = { 0, 0, 1, -1 };
  int n, m;

  bool inBound(int x, int y) {
    return x >= 0 && x < n && y >= 0 && y < m;
  }

  vector<vector<int>> bfs(int x, int y, vector<string>& maze) {
    vector<vector<int>> ret(n, vector<int>(m, -1));
    ret[x][y] = 0;
    queue<pair<int, int>> Q;
    Q.push({ x, y });
    while (!Q.empty()) {
      auto p = Q.front();
      Q.pop();
      int x = p.first, y = p.second;
      for (int k = 0; k < 4; k++) {
        int nx = x + dx[k], ny = y + dy[k];
        if (inBound(nx, ny) && maze[nx][ny] != '#' && ret[nx][ny] == -1) {
          ret[nx][ny] = ret[x][y] + 1;
          Q.push({ nx, ny });
        }
      }
    }
    return ret;
  }

  int minimalSteps(vector<string>& maze) {
    n = maze.size(), m = maze[0].size();
    // 机关 & 石头
    vector<pair<int, int>> buttons, stones;
    // 起点 & 终点
    int sx, sy, tx, ty;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        if (maze[i][j] == 'M') {
          buttons.push_back({ i, j });
        }
        if (maze[i][j] == 'O') {
          stones.push_back({ i, j });
        }
        if (maze[i][j] == 'S') {
          sx = i, sy = j;
        }
        if (maze[i][j] == 'T') {
          tx = i, ty = j;
        }
      }
    }
    int nb = buttons.size();
    int ns = stones.size();
    vector<vector<int>> start_dist = bfs(sx, sy, maze);

    // 边界情况：没有机关
    if (nb == 0) {
      return start_dist[tx][ty];
    }
    // 从某个机关到其他机关 / 起点与终点的最短距离。
    vector<vector<int>> dist(nb, vector<int>(nb + 2, -1));
    // 中间结果
    vector<vector<vector<int>>> dd(nb);
    for (int i = 0; i < nb; i++) {
      vector<vector<int>> d = bfs(buttons[i].first, buttons[i].second, maze);
      dd[i] = d;
      // 从某个点到终点不需要拿石头
      dist[i][nb + 1] = d[tx][ty];
    }

    for (int i = 0; i < nb; i++) {
      int tmp = -1;
      for (int k = 0; k < ns; k++) {
        int mid_x = stones[k].first, mid_y = stones[k].second;
        if (dd[i][mid_x][mid_y] != -1 && start_dist[mid_x][mid_y] != -1) {
          if (tmp == -1 || tmp > dd[i][mid_x][mid_y] + start_dist[mid_x][mid_y]) {
            tmp = dd[i][mid_x][mid_y] + start_dist[mid_x][mid_y];
          }
        }
      }
      dist[i][nb] = tmp;
      for (int j = i + 1; j < nb; j++) {
        int mn = -1;
        for (int k = 0; k < ns; k++) {
          int mid_x = stones[k].first, mid_y = stones[k].second;
          if (dd[i][mid_x][mid_y] != -1 && dd[j][mid_x][mid_y] != -1) {
            if (mn == -1 || mn > dd[i][mid_x][mid_y] + dd[j][mid_x][mid_y]) {
              mn = dd[i][mid_x][mid_y] + dd[j][mid_x][mid_y];
            }
          }
        }
        dist[i][j] = mn;
        dist[j][i] = mn;
      }
    }

    // 无法达成的情形
    for (int i = 0; i < nb; i++) {
      if (dist[i][nb] == -1 || dist[i][nb + 1] == -1) return -1;
    }

    // dp 数组， -1 代表没有遍历到
    vector<vector<int>> dp(1 << nb, vector<int>(nb, -1));
    for (int i = 0; i < nb; i++) {
      dp[1 << i][i] = dist[i][nb];
    }

    // 由于更新的状态都比未更新的大，所以直接从小到大遍历即可
    for (int mask = 1; mask < (1 << nb); mask++) {
      for (int i = 0; i < nb; i++) {
        // 当前 dp 是合法的
        if (mask & (1 << i)) {
          for (int j = 0; j < nb; j++) {
            // j 不在 mask 里
            if (!(mask & (1 << j))) {
              int next = mask | (1 << j);
              if (dp[next][j] == -1 || dp[next][j] > dp[mask][i] + dist[i][j]) {
                dp[next][j] = dp[mask][i] + dist[i][j];
              }
            }
          }
        }
      }
    }

    int ret = -1;
    int final_mask = (1 << nb) - 1;
    for (int i = 0; i < nb; i++) {
      if (ret == -1 || ret > dp[final_mask][i] + dist[i][nb + 1]) {
        ret = dp[final_mask][i] + dist[i][nb + 1];
      }
    }

    return ret;
  }
};

#endif // !_SOLUTION_