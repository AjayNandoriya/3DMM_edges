function [ R,t,s ] = EstimateSOPwithRefinement( xp,x )
%ESTIMATESOPWITHREFINEMENT Summary of this function goes here
%   Detailed explanation goes here

[ R,t,s ] = POS( xp,x );
b0 = Rts2b(double(R),double(t),double(s));

b = RefineOrthoCam( double(xp),double(x),b0 );

[R,t,s]=b2Rts(b);

end

