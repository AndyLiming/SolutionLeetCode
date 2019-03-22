#include "head.hpp"
#include "solution.h"
#include "test.h"

int main() {
  test te;
  solution solu;
  /*int l,m,n;
  vector<Interval> intervals, ans;
  cin >> l;
  for (int i = 0;i < l;++i) {
    cin >> m >> n;
    Interval a(m, n);
    intervals.push_back(a);
  }
  ans = s.merge(intervals);
  int l2 = ans.size();
  cout << "["<<endl;
  for (int i = 0;i < l2;++i) {
    cout << " [" << ans[i].start << "," << ans[i].end << "]" << endl;
  }
  cout << "]"<<endl;*/
  int n;
  cin >> n;
  vector<pair<int, int>> enves(n);
  for (int i = 0;i < n;++i) {
    cin >> enves[i].first >> enves[i].second;
  }
  cout << solu.maxEnvelopes(enves) << endl;
  //solu.wiggleSort(nums);
  //te.outputVector(nums);
  //cout << solu.countRangeSum(nums,l,u) << endl;
  //cout << solu.integerBreak(n) << endl;
  /*ListNode * head = new ListNode(1);
  ListNode *p = head;
  for (int i = 2;i <= 8;++i) {
    p->next = new ListNode(i);
    p = p->next;
  }
  ListNode *ans = solu.oddEvenList(head);
  p = ans;
  while (p != nullptr) {
    cout << p->val << " ";
    p = p->next;
  }*/
  //string s;
  //cin >> s;
  //cout << solu.reverseVowels(s) << endl;
  return 0;
}