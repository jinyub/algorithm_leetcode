#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <cstdlib>
#include <climits>
using namespace std;

#include <algorithm>
#include <cstring>

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

//求滑动窗口中的最大值，使用双端队列
vector<int> maxIntWindow(const vector<int>& num, unsigned int size) {
    deque<int> hd;//滑动窗口最大为size的大小,双端队列中存的是索引值
    vector<int> res;
    if(num.empty()||size==0||size>num.size())
        return res;
    hd.push_back(0);
    for (int i = 1; i < size-1; ++i) {
        if(num[i]<num[hd.back()]){
            hd.push_back(i);
        }else{
            while(!hd.empty()&&num[i]>=num[hd.back()]){
                hd.pop_back();
            }
            hd.push_back(i);
        }
    }
    for (int j = size-1; j < num.size(); ++j) {
        if(num[j]<num[hd.back()]){
            hd.push_back(j);
            if(hd.front()<j-size+1){
                hd.pop_front();
            }
            res.push_back(num[hd.front()]);
        }else{
            while(!hd.empty()&&num[j]>=num[hd.back()]){
                hd.pop_back();
            }
            hd.push_back(j);
            res.push_back(num[hd.front()]);
        }
    }
    return res;
}

/**
 * leetcode 56
 * 合并区间
 * 先排序，再判断
 * @param intervals
 * @return
 */

bool greater_first(const vector<int> & m1, const vector<int> & m2) {
    return m1[0] < m2[0];
}

vector<vector<int>> merge(vector<vector<int>>& intervals) {
    vector<vector<int>> res;
    if(intervals.empty())
        return res;
    //先对区间进行排序，使用STL自带的模板库
    sort(intervals.begin(),intervals.end(),greater_first);
    //再进行判断操作
    vector<int> temp={intervals[0][0],intervals[0][1]};
    for (int i = 1; i < intervals.size(); ++i) {
        if(intervals[i][0]>=temp[0]&&intervals[i][0]<=temp[1]){
            temp[0] = temp[0];
            temp[1] = temp[1]>intervals[i][1]?temp[1]:intervals[i][1];
        }else{
            res.push_back(temp);
            temp[0] = intervals[i][0];
            temp[1] = intervals[i][1];
        }
    }
    res.push_back(temp);
    return res;
}

/**
 * leetcode 75
 * 颜色分类
 * 使用常数空间的一趟排序算法
 * @param nums
 */
void sortColors(vector<int>& nums) {
    int num_one = 0;//1的个数
    if(nums.empty())
        return;
    int i=0, j=nums.size()-1,k=0;//i指向开头，j指向结尾
    while(nums[j]==2){
        j--;
    }
    for (k = 0; k <= j; ++k) {
        if(nums[k]==0){
            nums[i++] = nums[k];
        }else if(nums[k]==1){
            num_one++;
        }else{
            swap(nums[k],nums[j--]);
            if(nums[k]==0){
                nums[i++] = nums[k];
            }else if(nums[k]==1)
                num_one++;
            else {
                continue;
            }
        }
    }
    for (int l = i; l < i+num_one; ++l) {
        nums[l] = 1;
    }
}

void sortColors_r(vector<int>& nums) {
    if(nums.empty())
        return;
    int i=0,j=nums.size()-1;
    for (int curr = 0; curr <= j; ++curr) {
        if(nums[curr]==0){
            swap(nums[i++],nums[curr]);
        }else if(nums[curr]==2){
            swap(nums[curr--],nums[j--]);
        }
    }
}
/**
 * leetcode 5 最长回文子串
 * 从中间向两边扩散
 * 还有是无中心点的情况
 * @param s
 * @return
 */
string longestPalindrome(string s) {
    if(s.empty())
        return "";
    int len = s.length(),max_len=0;
    int max = 0;
    //判断全部相同的情况
    int start=0,end=0;
    for (int i = 0; i < len; ++i) {
        //有中心点情况
        int j=1,len1=0,len2=0;
        while((i-j)>=0&&(i+j)<=len&&s[i-j]==s[i+j]){
            j++;
            len1++;
        }
        len1 = 2*len1 + 1;

        //无中心点情况
        j=1;
        while((i-j)>=0&&(i+j-1)<=len&&s[i-j]==s[i+j-1]){
            j++;
            len2++;
        }
        len2 = 2*len2;

        len1 = len1 > len2 ? len1 : len2;
        if(len1>max){
            max = len1;
            start = i-len1/2;
            end = i+(len1-1)/2;
            max_len = len1;
        }
    }
    return s.substr(start,max_len);
}

/**
 * leetcode 137
 * 只出现一次的数字
 * @param nums
 * @return
 */
int singleNumber(vector<int>& nums) {

}

int __merge(vector<int>& nums, int l, int r) {
    int j=l;
    for (int i = l+1; i <= r; ++i) {
        if(nums[i]<nums[l]){
            j++;
            swap(nums[j],nums[i]);
        }
    }
    swap(nums[l],nums[j]);

    return j;
}

void __quick_sort(vector<int>& nums, int l, int r) {
    if(l>=r)
        return;
    int p = __merge(nums,l,r);
    __quick_sort(nums,l,p-1);
    __quick_sort(nums,p+1,r);
}

/**
 * 快速排序
 * @param nums
 */
void quick_sort(vector<int>& nums) {
    int len = nums.size();
    __quick_sort(nums,0,len-1);
}

bool __greater(const int & m1, const int & m2) {
    return m1 < m2;
}

/**
 * leetcode 15
 * 三数之和 a+b+c = 0
 * @param nums
 * @return
 */
vector<vector<int>> threeSum(vector<int>& nums) {
    vector<vector<int>> res;
    if(nums.empty())
        return res;
    int len = nums.size();
    int sum_t=0;
    //消除重复，先进行一个排序
    sort(nums.begin(),nums.end(),__greater);
    unordered_map<int,int> first;
    //固定一个数
    for (int i = 0; i < len-2; ++i) {
        if(first[nums[i]]){
            continue;
        }else{
            first[nums[i]] = 1;
        }
        sum_t = 0-nums[i];
        unordered_map<int,int> second;
        for (int j = i+1; j < len-1; ++j) {
            if(second[nums[j]]){
                continue;
            }else{
                second[nums[j]] = 1;
            }
            unordered_map<int,int> thrid;
            for (int k = j+1; k < len; ++k) {
                if(thrid[nums[k]]){
                    continue;
                }else{
                    thrid[nums[k]] = 1;
                }
                if(nums[j]+nums[k]==sum_t){
                    vector<int> temp = {nums[i],nums[j],nums[k]};
                    res.push_back(temp);
                    break;
                }else if(nums[j]+nums[k] > sum_t){
                    break;
                }
            }
        }
    }
    return res;
}

vector<vector<int>> threeSum_i (vector<int>& nums) {
    int target;
    vector<vector<int>> ans;
    sort(nums.begin(), nums.end());
    for (int i = 0; i < nums.size(); i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        if ((target = nums[i]) > 0) break;
        int l = i + 1, r = nums.size() - 1;
        while (l < r) {
            if (nums[l] + nums[r] + target < 0) ++l;
            else if (nums[l] + nums[r] + target > 0) --r;
            else {
                ans.push_back({target, nums[l], nums[r]});
                ++l, --r;
                while (l < r && nums[l] == nums[l - 1]) ++l;
                while (l < r && nums[r] == nums[r + 1]) --r;
            }
        }
    }
    return ans;
}


//美团笔试
int mid_search(vector<int> a1,vector<int> a2,int f_1,int f_2,int s_1,int s_2){
    if(f_1==f_2){
        return a1[f_1];
    }
    if(s_1==s_2){
        return a2[s_1];
    }
    int mid1 = (f_2+f_1)/2;
    int mid2 = (s_2+s_1)/2;
    if(a1[mid1]>a2[mid2]){
        return mid_search(a1,a2,f_1,mid1,mid2,s_2);
    }else if(a1[mid1]<a2[mid2]){
        return mid_search(a1,a2,mid1,f_2,s_1,mid2);
    } else
        return a1[mid1];
}


//字符串的转置
void reverse(vector<char>& s, int size) {
    int len = s.size();
    int mid = len/2;

    for (int i = 0; i < mid; ++i) {
        swap(s[i],s[len-1-i]);
    }
    int mid1 = (len-size)/2;
    int start = len-size-1;
    for (int j = 0; j < mid1; ++j) {
        swap(s[j],s[start-j]);
    }
    int mid2 = size/2;
    int j=0;
    for (int k = start+1; k < start+1+mid2; ++k) {
        swap(s[k],s[len-1-j]);
        j++;
    }
}

/**
 * 正则表达式匹配
 * 'aaa' 与 'ab*ac*a'匹配
 * @param str
 * @param pattern
 * @return
 */
bool match_u(char* str, char* pattern, int a, int b) {
    if(pattern[b]=='\0'){
        if(str[a]=='\0')
            return true;
        else
            return false;
    }else if(pattern[b]=='.'){
        if(pattern[b+1]=='*'){
            if(str[a]=='\0') {
                if(a==0)
                    return true;
                else
                    return false;
            }
            return match_u(str,pattern,a+1,b) || match_u(str,pattern,a+1,b+2) || match_u(str,pattern,a,b+2);
        } else {
            if(str[a]=='\0')
                return false;
            return match_u(str, pattern, a + 1, b + 1);
        }
    }else {
        if(pattern[b]==str[a]) {
            if(pattern[b+1]=='*'){
                return match_u(str,pattern,a+1,b) || match_u(str,pattern,a+1,b+2) || match_u(str,pattern,a,b+2);
            }
            return match_u(str, pattern, a + 1, b + 1);
        }
        else if(str[a]=='\0'){
            if(pattern[b+1]=='*'){
                return match_u(str,pattern,a,b+2);
            }
        }
        else if(str[a]!='\0'){
            if(pattern[b+1]=='*'){
                return match_u(str,pattern,a,b+2);
            }
        }
    }
    return false;
}

bool match(char* str, char* pattern)
{
    return match_u(str,pattern,0,0);
}


int main() {
    char a[] = {'a','a','a','\0'};
    char b[] = {'a','*','a','\0'};
    char c[] = {'a','b','*','a','*','c','*','a','\0'};
    char d[] = {'a','\0'};
    char e[] = {'a','b','*','a','\0'};
    char f[] = {'b','c','b','b','a','b','a','b','\0'};
    char f1[] = {'.','*','a','*','a','\0'};
    char g[] = {'\0'};
    char g1[] = {'c','*','\0'};
    char x[] = {'b','b','b','b','a','\0'};
    char x1[] = {'.','*','a','*','a','\0'};
    cout << match(g,g1) << endl;
    cout << match(f,f1) << endl;
    cout << match(a,b) << endl;
    cout << match(a,c) << endl;
    cout << match(d,g) << endl;
    cout << match(x,x1) << endl;

    return 0;
}