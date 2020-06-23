#include "head.hpp"
#include "solution.h"
#include "test.h"



int main() {
  test te;
  solution solu;

  //input List
  /*for (int i = 0;i < n;++i) {
    int tmp;
    cin >> tmp;
    p->next = new ListNode(tmp);
    p = p->next;
  }*/
  //output List
  /*p = ans;
  while (p != nullptr) {
    cout << p->val << " ";
    p = p->next;
  }
  cout << endl;*/

  //int n;
  //cin >> n;
  //vector<int>nums(n);
  //for (int i = 0;i < n;++i) cin >> nums[i];
  //cout << solu.longestConsecutive(nums) << endl;
  //TreeNode *root = new TreeNode(10);
  //root->left = new TreeNode(5);root->right = new TreeNode(-3);
  //root->left->left = new TreeNode(3);root->left->right = new TreeNode(2);root->right->right = new TreeNode(11);

  //vector<vector<int>> input;
  //te.inputVectorVec(input);
  //solu.gameOfLife(input);
  //te.outputVectorVec(input);

  string s = "mississippi";
  string p = "mis*is*p*.";
  cout << isMatch(s, p) << endl;

  return 0;
}