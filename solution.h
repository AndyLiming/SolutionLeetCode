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

  bool isPalindrome(ListNode* head);
  //234. Palindrome Linked List: Given a singly linked list, determine if it is a palindrome.

  TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
  //235. Lowest Common Ancestor of a Binary Search Tree:¡¡Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

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
  //338. Counting Bits: Given a non negative integer number num. For every numbers i in the range 0 ¡Ü i ¡Ü num calculate the number of 1's in their binary representation and return them as an array.

  int maxProfit(vector<int>& prices);
  //309. Best Time to Buy and Sell Stock with Cooldown: Say you have an array for which the ith element is the price of a given stock on day i.
  //Design an algorithm to find the maximum profit.You may complete as many transactions as you like(ie, buy one and sell one share of the stock multiple times) with the following restrictions :

  int countRangeSum(vector<int>& nums, int lower, int upper);
  //327. Count of Range Sum: Given an integer array nums, return the number of range sums that lie in [lower, upper] inclusive.
  //Range sum S(i, j) is defined as the sum of the elements in nums between indices i and j(i ¡Ü j), inclusive.

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
  //357. Count Numbers with Unique Digits: Given a non-negative integer n, count all numbers with unique digits, x, where 0 ¡Ü x < 10n.

  int maxSumSubmatrix(vector<vector<int>>& matrix, int k);
  //363. Max Sum of Rectangle No Larger Than K: Given a non-empty 2D matrix matrix and an integer k, find the max sum of a rectangle in the matrix such that its sum is no larger than k.

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
#endif // !_SOLUTION_

