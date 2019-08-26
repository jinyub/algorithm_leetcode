#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <stdlib.h>
using namespace std;

#include <algorithm>
#include <string.h>



//找滑动窗口中的最大值
//这种思路相当于求固定size中的最大值，时间复杂度为O(n*size)
vector<int> maxInWindows(const vector<int>& num, unsigned int size)
{
    vector<int> res;
    if(num.empty()||size==0)
        return res;
    int max=INT_MIN,start=1,max_index=0;
    for (int j = 0; j < size; ++j) {
        if(num[j]>max){
            max = num[j];
            max_index = j;
        }
    }
    res.push_back(max);
    for (int i = start; i < num.size()-size+1; ++i) {
        if(max_index!=i-1){
            if(num[i+size-1]>max){
                max = num[i+size-1];
                max_index = i+size-1;
            }
            res.push_back(max);
        }else{
            max = INT_MIN;
            for (int j = i; j < i+size-1; ++j) {
                if(num[j]>max){
                    max = num[j];
                    max_index = j;
                }
            }
            if(num[i+size-1]>max){
                max = num[i+size-1];
                max_index = i+size-1;
            }
            res.push_back(max);
        }
    }
    return res;
}

//京东笔试题
//拼接迷宫
bool is_migong(vector<char> lj,int start,int row, int col) {
    lj[start] = '#';
    //四个方向
    int row_index = start/(col*3);
    int col_index = start/(col*3);
    if(row_index==0||col_index==0||row_index==3*row-1||col_index==3*col-1)
        return true;
    int new_start;
    bool res = false;
    //向上
    row_index--;
    new_start = row_index*col*3+col_index;
    if(row_index>=0&&lj[new_start]=='.'){
        res = is_migong(lj,new_start,row,col);
    }
    //向左
    row_index++;
    col_index--;
    new_start = row_index*col*3+col_index;
    if(!res&&col_index>=0&&lj[new_start]=='.'){
        res = is_migong(lj,new_start,row,col);
    }
    //向下
    col_index++;
    row_index++;
    new_start = row_index*col*3+col_index;
    if(!res&&row_index<row&&lj[new_start]=='.'){
        res = is_migong(lj,new_start,row,col);
    }
    //向右
    row_index--;
    col_index++;
    new_start = row_index*col*3+col_index;
    if(!res&&col_index<col&&lj[new_start]=='.'){
        res = is_migong(lj,new_start,row,col);
    }

    return res;
}


int main() {
    int t;
    cin >> t;
    while(t--) {
        int n, m;
        cin >> n >> m;
        int len = n * m, count = 0, start = 0;
        vector<char> lj;
        char temp;
        while (len--) {
            cin >> temp;
            if (temp == 'S')
                start = count;
            lj.push_back(temp);
            count++;
        }
        //构建一个1*3的大块
        vector<char> pj;
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < m; ++j) {
                    temp = lj[i * m + j];
                    pj.push_back(temp);
                }
            }
        }
        //构建出剩余的块
        vector<char> fin;
        for (int l = 0; l < 3; ++l) {
            for (int i = 0; i < pj.size(); ++i) {
                if (l == 1) {
                    int row = start/m;
                    int col = start%m;

                    if(i == (row)*3*m+col+m){
                        temp = 'S';
                        fin.push_back(temp);
                    } else{
                        if(pj[i]=='S')
                            fin.push_back('.');
                        else
                            fin.push_back(pj[i]);
                    }

                } else {
                    if(pj[i]=='S')
                        fin.push_back('.');
                    else
                        fin.push_back(pj[i]);
                }
            }
        }

        bool res = is_migong(fin,4*n*m-2+start,n,m);
        if(res)
            cout << "Yes" << endl;
        else
            cout << "No" << endl;
        for (int k = 0; k < fin.size(); ++k) {
            cout << fin[k];
        }
        //cout << 4*n*m-2+start << endl;
    }
    return 0;
}