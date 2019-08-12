//
// Created by xiping on 2019-02-17.
// leetcode 122. 贪心算法，只要后面的

#ifndef LEETCODE_MAXPROFIT_H
#define LEETCODE_MAXPROFIT_H

#include <vector>
using namespace std;


int maxProfit(vector<int>& prices) {
    if(prices.empty())
        return 0;
    int buy_in_price = prices[0];
    int sold_out_price = prices[0];
    int profit = 0;
    for (int k = 1; k < prices.size(); ++k) {
        if(prices[k] > sold_out_price){
            sold_out_price = prices[k];
        }else{
            profit += sold_out_price - buy_in_price;
            buy_in_price = sold_out_price = prices[k];
        }
    }
    profit += sold_out_price - buy_in_price;
    return profit;
}

#endif //LEETCODE_MAXPROFIT_H
