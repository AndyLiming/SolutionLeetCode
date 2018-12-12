#include "head.hpp"
#include "solution.h"
#include "test.h"

int main() {
  test t;
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
  //int k;
  vector<int> nums;
  //cin >> k;
  t.inputVector(nums);
  //solu.rotate(nums, k);
  //t.outputVector(nums);
  cout << solu.rob(nums) << endl;
  return 0;
}