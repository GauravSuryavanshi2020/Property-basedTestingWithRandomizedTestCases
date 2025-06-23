#include <rapidcheck.h>
#include <ql/quantlib.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/pricingengines/blackformula.hpp>

using namespace QuantLib;

class QuantLibPropertyTests {
public:
    // Property 1: Interest Rate Compounding Consistency
    static void testCompoundingConsistency() {
        rc::check("Compounding frequency consistency", []() {
            auto rate = *rc::gen::inRange<Rate>(0.01, 0.30);
            auto time = *rc::gen::inRange<Time>(0.1, 10.0);
            
            // Test different compounding frequencies
            InterestRate continuousRate(rate, Actual365Fixed(), Continuous);
            InterestRate annualRate(rate, Actual365Fixed(), Annual);
            
            Real continuousDF = continuousRate.discountFactor(time);
            Real annualDF = annualRate.discountFactor(time);
            
            // Property: Discount factors should be positive and <= 1
            RC_ASSERT(continuousDF > 0.0 && continuousDF <= 1.0);
            RC_ASSERT(annualDF > 0.0 && annualDF <= 1.0);
            
            // Property: Higher compounding frequency should give lower discount factor
            // (for positive rates and positive time)
            if (rate > 0.0 && time > 0.0) {
                RC_ASSERT(continuousDF <= annualDF);
            }
        });
    }
    
    // Property 2: Black-Scholes Formula Properties
    static void testBlackScholesProperties() {
        rc::check("Black-Scholes option pricing properties", []() {
            auto spot = *rc::gen::inRange<Real>(10.0, 200.0);
            auto strike = *rc::gen::inRange<Real>(10.0, 200.0);
            auto riskFreeRate = *rc::gen::inRange<Rate>(0.01, 0.20);
            auto volatility = *rc::gen::inRange<Volatility>(0.05, 1.0);
            auto timeToExpiry = *rc::gen::inRange<Time>(0.01, 5.0);
            
            Real callPrice = blackFormula(Option::Call, strike, spot, 
                                        std::sqrt(volatility * volatility * timeToExpiry),
                                        std::exp(-riskFreeRate * timeToExpiry));
            
            Real putPrice = blackFormula(Option::Put, strike, spot,
                                       std::sqrt(volatility * volatility * timeToExpiry),
                                       std::exp(-riskFreeRate * timeToExpiry));
            
            // Property 1: Option prices are non-negative
            RC_ASSERT(callPrice >= 0.0);
            RC_ASSERT(putPrice >= 0.0);
            
            // Property 2: Call option intrinsic value
            RC_ASSERT(callPrice >= std::max(0.0, spot - strike * std::exp(-riskFreeRate * timeToExpiry)));
            
            // Property 3: Put option intrinsic value
            RC_ASSERT(putPrice >= std::max(0.0, strike * std::exp(-riskFreeRate * timeToExpiry) - spot));
            
            // Property 4: Put-Call Parity (approximately)
            Real putCallDifference = callPrice - putPrice - spot + strike * std::exp(-riskFreeRate * timeToExpiry);
            RC_ASSERT(std::abs(putCallDifference) < 1e-10);
        });
    }
    
    // Property 3: Yield Curve Interpolation Properties
    static void testYieldCurveProperties() {
        rc::check("Yield curve interpolation properties", []() {
            // Generate random yield curve data
            auto numPoints = *rc::gen::inRange(3, 10);
            std::vector<Date> dates;
            std::vector<Rate> rates;
            
            Date referenceDate(15, July, 2020);
            dates.push_back(referenceDate);
            rates.push_back(*rc::gen::inRange<Rate>(0.01, 0.05));
            
            for (int i = 1; i < numPoints; ++i) {
                dates.push_back(referenceDate + Period(i * 365, Days));
                rates.push_back(*rc::gen::inRange<Rate>(0.01, 0.08));
            }
            
            // Create yield curve
            DayCounter dayCounter = Actual365Fixed();
            Handle<YieldTermStructure> yieldCurve(
                boost::make_shared<ZeroCurve>(dates, rates, dayCounter, 
                                            TARGET(), Linear())
            );
            
            // Property 1: Discount factors are monotonically decreasing
            for (size_t i = 1; i < dates.size(); ++i) {
                Real df1 = yieldCurve->discount(dates[i-1]);
                Real df2 = yieldCurve->discount(dates[i]);
                RC_ASSERT(df2 <= df1); // Later dates should have lower discount factors
            }
            
            // Property 2: All discount factors are between 0 and 1
            for (const auto& date : dates) {
                Real df = yieldCurve->discount(date);
                RC_ASSERT(df > 0.0 && df <= 1.0);
            }
        });
    }
    
    // Property 4: Date Arithmetic Properties
    static void testDateArithmetic() {
        rc::check("Date arithmetic properties", []() {
            auto year = *rc::gen::inRange(1900, 2100);
            auto month = *rc::gen::inRange(1, 12);
            auto day = *rc::gen::inRange(1, 28); // Safe day range
            auto addDays = *rc::gen::inRange(1, 3650); // Up to 10 years
            
            Date originalDate(day, static_cast<Month>(month), year);
            Date futureDate = originalDate + Period(addDays, Days);
            
            // Property 1: Adding positive days results in a later date
            RC_ASSERT(futureDate > originalDate);
            
            // Property 2: The difference should equal the added days
            RC_ASSERT(futureDate - originalDate == addDays);
            
            // Property 3: Subtracting the same period gives original date
            Date backToOriginal = futureDate - Period(addDays, Days);
            RC_ASSERT(backToOriginal == originalDate);
        });
    }
    
    // Property 5: Calendar Business Day Properties
    static void testCalendarProperties() {
        rc::check("Calendar business day properties", []() {
            TARGET calendar; // European Central Bank calendar
            
            auto year = *rc::gen::inRange(2020, 2030);
            auto month = *rc::gen::inRange(1, 12);
            auto day = *rc::gen::inRange(1, 28);
            
            Date testDate(day, static_cast<Month>(month), year);
            
            // Property 1: adjust() should return a business day
            Date adjustedDate = calendar.adjust(testDate);
            RC_ASSERT(calendar.isBusinessDay(adjustedDate));
            
            // Property 2: If date is already a business day, adjust() returns same date
            if (calendar.isBusinessDay(testDate)) {
                RC_ASSERT(adjustedDate == testDate);
            }
            
            // Property 3: advance() by 0 days returns same date (if business day)
            if (calendar.isBusinessDay(testDate)) {
                Date sameDate = calendar.advance(testDate, 0, Days);
                RC_ASSERT(sameDate == testDate);
            }
        });
    }
};

// Test runner
int main() {
    try {
        QuantLibPropertyTests::testCompoundingConsistency();
        QuantLibPropertyTests::testBlackScholesProperties();
        QuantLibPropertyTests::testYieldCurveProperties();
        QuantLibPropertyTests::testDateArithmetic();
        QuantLibPropertyTests::testCalendarProperties();
        
        std::cout << "All property tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Property test failed: " << e.what() << std::endl;
        return 1;
    }
}