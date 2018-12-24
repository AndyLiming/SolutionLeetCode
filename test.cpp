#include "head.hpp"
#include "test.h"

template<class T>
void test::inputVector(vector<T>& input) {
  T tmp;
  if (!input.empty()) {
    input.clear();
  }
  while (cin >> tmp) {
    input.push_back(tmp);
  }
}

template<class T >
void test::outputVector(vector<T> output){
  cout << "[ ";
  for (int i = 0;i < output.size();++i) {
    cout << output[i];
    cout << ", ";
  }
  cout << "]" << endl;
}

template<class T>
void test::outputVectorVec(vector<vector<T>> output){
  cout << "[ " << endl;
  for (int i = 0;i < output.size();++i) {
    outputVector(output[i]);
  }
  cout << "]" << endl;
}
