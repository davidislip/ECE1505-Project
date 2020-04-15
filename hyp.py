
def get_list_of_trade_times(theta_val, ARL_target, n, limit=2*int(len(R)/n)):
    
    t_mid = n+1
    t_mids = []
    Ts = []
    ARLs = []
    balance = []
    k = 0
    while t_mid <= len(R)-n - 1 and k < limit: #k < 2*int(len(R)/n):
        t_mids.append(t_mid)
        t_now = t_mid+n
        #single step change point algorithm
        [p1,p2, gamma_1, gamma_2, prob, Q] = robust_hypothesis(t_mid, n, theta_val, R)
        signal = p1.value - p2.value
        signal[np.abs(signal) < 10**(-3)*np.max(signal)] = 0
        phi = np.sign(signal)
        
        for b in range(1,n):
            arl = calculate_ARL(p1, p2, phi, b, plots = False)
            if arl > ARL_target:
                break 
        T, S = detect_change(phi, b)
    
        if T == 2*n-1: #no change in this sample
            t_mid = t_now + n
            #no rebalance
        elif T > n:
            t_mid = t_mid + T - n
            balance.append(t_now)
            Ts.append(T)
            ARLs.append(arl)
        elif T <= n:
            t_mid = t_mid + T
            balance.append(t_now)
            Ts.append(T)
            ARLs.append(arl)
        k = k + 1
        if k == limit:
            print("Too many rebalancing points")
    return {"times":balance, "mid_points": t_mids, "Ts": Ts, "ARL":ARLs}
