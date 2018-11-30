%输入已达标的两路信号，计算ptt并返回
function ptt = calculate_ptt(window_heart_rate,window_ECG)

    ECG_diff = diff(window_ECG,1,2);
    
    % ECG_diff 是 ECG的一阶导数，往往会比ECG少一个点
    if length(ECG_diff) ~= length(window_heart_rate)
        ECG_diff = [[0;0;0;0;0],ECG_diff];
    end

    %数据归一化到区间[0,1]
%     nor_window_heart_rate = mapminmax(window_heart_rate,0,1);
%     nor_window_ECG = mapminmax(window_ECG,0,1);
%     nor_ECG_diff = mapminmax(ECG_diff,0,1);
    
    nor_window_heart_rate = window_heart_rate;
    nor_window_ECG = window_ECG;
    nor_ECG_diff = ECG_diff;
    
    f = figure;
    set(f,'position',[500 100 300 600]);

    X = 1:length(nor_window_heart_rate);
    Y1 = nor_window_heart_rate(1,:);
    Y2 = nor_window_heart_rate(2,:);
    Y3 = nor_window_heart_rate(3,:);
    Y4 = nor_window_heart_rate(4,:);
    Y5 = nor_window_heart_rate(5,:);

    Z1 = nor_window_ECG(1,:);
    Z2 = nor_window_ECG(2,:);
    Z3 = nor_window_ECG(3,:);
    Z4 = nor_window_ECG(4,:);
    Z5 = nor_window_ECG(5,:);

    W1 = nor_ECG_diff(1,:);
    W2 = nor_ECG_diff(2,:);
    W3 = nor_ECG_diff(3,:);
    W4 = nor_ECG_diff(4,:);
    W5 = nor_ECG_diff(5,:);

    % 计算每个窗口的ptt
    subplot(5,1,1);
    plot(X, Y1, 'b', X, Z1,'r', X, W1, ':k');
    [max_Y1,index_Y1] = max(Y1);
    [max_W1,index_W1] = max(W1);
    strY1=['(',num2str(X(index_Y1)),',', num2str(max_Y1),')'];
    text(index_Y1,max_Y1,strY1);
    strW1=['(',num2str(X(index_W1)),',', num2str(max_W1),')'];
    text(index_W1,max_W1,strW1);
    PTT1 = index_W1 - index_Y1;

    subplot(5,1,2);
    plot(X, Y2, 'b', X, Z2, 'r', X, W2, ':k');
    [max_Y2,index_Y2] = max(Y2);
    [max_W2,index_W2] = max(W2);
    strY2=['(',num2str(X(index_Y2)),',', num2str(max_Y2),')'];
    text(index_Y2,max_Y2,strY2);
    strW2=['(',num2str(X(index_W2)),',', num2str(max_W2),')'];
    text(index_W2,max_W2,strW2);
    PTT2 = index_W2 - index_Y2;

    subplot(5,1,3);
    plot(X, Y3, 'b', X, Z3, 'r', X, W3, ':k');
    [max_Y3,index_Y3] = max(Y3);
    [max_W3,index_W3] = max(W3);
    strY3=['(',num2str(X(index_Y3)),',', num2str(max_Y3),')'];
    text(index_Y3,max_Y3,strY3);
    strW3=['(',num2str(X(index_W3)),',', num2str(max_W3),')'];
    text(index_W3,max_W3,strW3);
    PTT3 = index_W3 - index_Y3;

    subplot(5,1,4);
    plot(X, Y4, 'b', X, Z4, 'r', X, W4, ':k');
    [max_Y4,index_Y4] = max(Y4);
    [max_W4,index_W4] = max(W4);
    strY4=['(',num2str(X(index_Y4)),',', num2str(max_Y4),')'];
    text(index_Y4,max_Y4,strY4);
    strW4=['(',num2str(X(index_W4)),',', num2str(max_W4),')'];
    text(index_W4,max_W4,strW4);
    PTT4 = index_W4 - index_Y4;

    subplot(5,1,5);
    plot(X, Y5, 'b', X, Z5, 'r', X, W5, ':k');
    [max_Y5,index_Y5] = max(Y5);
    [max_W5,index_W5] = max(W5);
    strY5=['(',num2str(X(index_Y5)),',', num2str(max_Y5),')'];
    text(index_Y5,max_Y5,strY5);
    strW5=['(',num2str(X(index_W5)),',', num2str(max_W5),')'];
    text(index_W5,max_W5,strW5);
    PTT5 = index_W5 - index_Y5;

    PTT = [PTT1, PTT2, PTT3, PTT4, PTT5];
    % 求平均值，将ptt转化成以ms为单位的数
    ptt = mean(PTT) * 20;
end




