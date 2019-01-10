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
  //vector<int> nums;
  //te.inputVector(nums);
  //cout << solu.rob2(nums) << endl;

  string s;
  cin >> s;
  //cout << solu.calculate(s) << endl;
  //int k;
  //cin >> k;
  //cout << solu.isPowerOfTwo(k) << endl;
  //vector<int> ans = solu.productExceptSelf(nums);
  //te.outputVector(ans);
  vector<int> ans = solu.diffWaysToCompute(s);
  te.outputVector(ans);
  return 0;
}