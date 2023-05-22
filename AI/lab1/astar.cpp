#include <bits/stdc++.h>
#define pr pair<int,int>
#define LL unsigned long long

using namespace std;
#define N 3//N*N����
vector<int> deconvert(LL state) {// ��״̬ѹ��ֵ(64λ�޷�������)ת��Ϊ��������
    vector<int>v;
    for (int i = 0; i < N * N; i++) {
        v.push_back(state % (N * N));
        state /= N * N;
    }
    return v;
}
LL convert(vector<int>v) {//����������ת��Ϊ״̬ѹ��ֵ

    LL ans = 0;
    for (int i = N * N - 1; i >= 0; i--) {
        ans = ans * N * N + v[i];
    }
    return ans;
}
void print(LL state) {
    vector<int>v = deconvert(state);
    for (int i = 0; i < N * N; i++) {
        printf("%d%c", v[i], i % N == N - 1 ? '\n' : ' ');
    }
}

LL read() {//����9�����ֲ�������״̬��ѹ��ֵ
    vector<int>v(N * N);
    for (auto& x : v)scanf("%d", &x);
    return convert(v);
}
int fx[] = { 1,-1,0,0 };//�ĸ������������ϵ�λ��(��,��,��,��)
int fy[] = { 0,0,1,-1 };
vector<LL>getnextstate(LL state) {//��ȡ��ǰ״̬�ܹ�ת�Ƶ�����һ��״̬
    auto v = deconvert(state);
    vector<LL>ans;
    for (int i = 0; i < N * N; i++) {
        if (v[i] == 0) {
            int x = i / N, y = i % N;//��ȡ��ǰ״̬��3x3�����ϵ�λ��
            for (int k = 0; k < 4; k++) {
                int nx = x + fx[k];//��ȡ��4�������ϵ���һ��λ��
                int ny = y + fy[k];
                if (nx >= 0 && nx < N && ny >= 0 && ny < N) {//�����һ��λ��û��Խ��
                    swap(v[i], v[nx * N + ny]);//������Ӧֵ
                    ans.push_back(convert(v));//����ǰ״̬��ѹ��ֵ������
                    swap(v[i], v[nx * N + ny]);//�ָ�
                }
            }
            break;
        }
    }
    return ans;

}
int target[N * N];//target[i]��ʾ����iӦȥ��λ��
int getval(vector<int>v) {//��ֵ����
    int ans = 0;
    for (int i = 0; i < N * N; i++) {
        int x = target[v[i]];
        ans += abs(x % N - i % N) + abs(x / N - i / N);
        //ans += (i != x);
    }

    return ans;
}
map<LL, pair<int, LL>>pre;//{state,{h(s)+g(s),prestate}},����״̬�ļ�ֵ��ǰ��
set<LL>use;//close_list
void init(vector<int>v) {//��ʼ��Ŀ��״̬
    for (int i = 0; i < N * N; i++) {
        target[v[i]] = i;
    }

}
void solve() {
    LL fans = 0;

    auto s = deconvert(read());//��ȡ��ʼ״̬
    init(deconvert(read()));
    priority_queue<pair<int, LL>> q;
    if (getval(s) == 0) {//�����ʼ״̬��ֵΪ0, ��ʾ����Ŀ��״̬
        fans = convert(s);
        printf("input is final\n");
    }
    q.push({ -getval(s),convert(s) });//����ʼ״̬����open_list
    while ((!q.empty()) && (!fans)) {//������ȶ��в�Ϊ����δ�ҵ���
        int val = -q.top().first;//��ǰ���ż�ֵ
        auto ns = q.top().second;//��ǰ״̬��ѹ��ֵ
        q.pop();
        if (use.find(ns) != use.end()) {//�����״̬����close_list��,����
            continue;
        }
        use.insert(ns);//����close_list
        for (auto x : getnextstate(ns)) {
            if (use.find(x) != use.end())continue; //�����״̬����close_list��,����
            int gvx = getval(deconvert(x));//��ȡ��״̬��g
            int nv = val - getval(deconvert(ns)) + gvx + 1;//�����ֵ����f
            if (pre.find(x) == pre.end() || (*pre.find(x)).second.first > nv) {//�����״̬��ǰ������һ��ת��ʱ��ֵ��������,����ת��
                q.push({ -nv,x });//����open_list
                pre[x] = { nv,ns };//��¼ǰ��
            }
            if (gvx == 0) {//���gΪ0,�ѵ����յ�,��¼��
                fans = x;
                break;
            }
        }
    }
    if (!fans) {//δ�ҵ��յ�
        printf("ans not found\n");
        return;
    }
    else {
        printf("ans found: \n");
    }
    vector<LL>ans;
    ans.push_back(fans);
    while (pre.find(fans) != pre.end()) {//����ǰ���ҵ�ת��·��
        fans = pre[fans].second;
        ans.push_back(fans);
        //print(fans);
    }
    for (int i = ans.size() - 1; i >= 0; i--) {//���ת��·��
        print(ans[i]);
        printf("___step %d____\n", ans.size() - i);
    }
}
int main() {

    solve();

}
