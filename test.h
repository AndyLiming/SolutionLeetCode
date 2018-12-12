#ifndef _TEST_
#define _TEST_


class test {
public:
    //input
  template<class T>
  void inputVector(vector<T> &input) {
    T tmp;
    if (!input.empty()) {
      input.clear();
    }
    while (cin >> tmp) {
      input.push_back(tmp);
    }
  }
  template<class T>
  void inputVectorVec(vector<vector<T>> &input) {
    /*int n, m;*/
    int n;
    cin >> n;
    //cin >> m;
    vector<T> inputr;
    T tmp;
    for (int i = 0;i < n;++i) {
      inputr.clear();
      /*for (int j = 0;j < m;++j) {
        cin >> tmp;
        inputr.push_back(tmp);
      }*/
      while(cin>>tmp) inputr.push_back(tmp);
      input.push_back(inputr);

    }
  }
    //output
  template<class T>
  void outputVector(vector<T> output) {
    cout << "[";
    if (output.size() == 0) {
      cout << " ]" << endl;;
      return;
    }
    for (int i = 0;i < output.size()-1;++i) {
      cout << output[i];
      cout << ", ";
    }
    cout << output[output.size() - 1];
    cout << "]" << endl;
  }

  template<class T>
  void outputVectorVec(vector<vector<T>> output) {
    cout << "[ " << endl;
    for (int i = 0;i < output.size();++i) {
      outputVector(output[i]);
    }
    cout << "]" << endl;
  }
};
#endif
