// CMakeLists.txt addition for RapidCheck integration
find_package(PkgConfig REQUIRED)
pkg_check_modules(RAPIDCHECK REQUIRED rapidcheck)

# Add to your test executable
target_link_libraries(your_test_executable ${RAPIDCHECK_LIBRARIES})
target_include_directories(your_test_executable PRIVATE ${RAPIDCHECK_INCLUDE_DIRS})

// Basic Property Test Setup for QuantLib
#include <rapidcheck.h>
#include <ql/quantlib.hpp>
#include <iostream>

using namespace QuantLib;

// Custom generators for QuantLib types
namespace rc {
    template<>
    struct Arbitrary<Rate> {
        static Gen<Rate> arbitrary() {
            return gen::inRange<Rate>(0.0, 0.50); // 0% to 50% rates
        }
    };
    
    template<>
    struct Arbitrary<Time> {
        static Gen<Time> arbitrary() {
            return gen::inRange<Time>(0.1, 30.0); // 0.1 to 30 years
        }
    };
    
    template<>
    struct Arbitrary<Date> {
        static Gen<Date> arbitrary() {
            return gen::map(gen::inRange(1, 10000), [](int days) {
                return Date(1, January, 2020) + Period(days, Days);
            });
        }
    };
}